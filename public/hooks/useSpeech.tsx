import { AutoTokenizer } from "@huggingface/transformers";
import { getModelFile } from "@huggingface/transformers/utils/hub.js";
import * as ort from "onnxruntime-web/webgpu";

export async function recognizeSpeechFromAudio(
  audio: Float32Array
): Promise<string> {
  const modelRepo = "csukuangfj/sherpa-onnx-moonshine-base-en-int8";
  const tokenizerRepo = "UsefulSensors/moonshine-base";

  // Load all models first
  const preprocessModel = await getModelFile(modelRepo, "preprocess.onnx");
  const encodeModel = await getModelFile(modelRepo, "encode.int8.onnx");
  const uncachedDecodeModel = await getModelFile(
    modelRepo,
    "uncached_decode.int8.onnx"
  );
  const cachedDecodeModel = await getModelFile(
    modelRepo,
    "cached_decode.int8.onnx"
  );

  // Create all sessions
  const sessionOptions = { executionProviders: ["wasm"] };

  const preprocess = await ort.InferenceSession.create(
    preprocessModel,
    sessionOptions
  );
  const encode = await ort.InferenceSession.create(encodeModel, sessionOptions);
  const uncachedDecode = await ort.InferenceSession.create(
    uncachedDecodeModel,
    sessionOptions
  );
  const cachedDecode = await ort.InferenceSession.create(
    cachedDecodeModel,
    sessionOptions
  );

  // Load tokenizer
  const tokenizer = await AutoTokenizer.from_pretrained(tokenizerRepo);

  const sos_id = 1; // <s>
  const eos_id = 2; // </s>

  // Preprocess audio
  const audioTensor = new ort.Tensor("float32", audio, [1, audio.length]);
  const preprocessOutput = await preprocess.run({
    [preprocess.inputNames[0]]: audioTensor,
  });
  const features = preprocessOutput[preprocess.outputNames[0]];

  // Run encoder
  const featuresLen = new ort.Tensor(
    "int32",
    new Int32Array([features.dims[1]]),
    [1]
  );
  const encodeOutput = await encode.run({
    [encode.inputNames[0]]: features,
    [encode.inputNames[1]]: featuresLen,
  });
  const encoderOut = encodeOutput[encode.outputNames[0]];

  // First decode step
  const tokenTensor = new ort.Tensor("int32", new Int32Array([sos_id]), [1, 1]);
  const tokenLenTensor = new ort.Tensor("int32", new Int32Array([1]), [1]);

  const firstDecodeOutput = await uncachedDecode.run({
    [uncachedDecode.inputNames[0]]: tokenTensor,
    [uncachedDecode.inputNames[1]]: encoderOut,
    [uncachedDecode.inputNames[2]]: tokenLenTensor,
  });

  let logits = firstDecodeOutput["reversible_embedding"];
  let states = {};
  for (const key of Object.keys(firstDecodeOutput)) {
    if (key !== "reversible_embedding") {
      states[key] = firstDecodeOutput[key];
    }
  }

  const maxLen = Math.floor((audio.length / 16000) * 6);
  const generated_tokens: number[] = [];

  // Decode loop
  for (let i = 0; i < maxLen; i++) {
    const logitsArray = new Float32Array(logits.data);
    const token = Array.from(logitsArray.keys()).reduce(
      (maxIdx, curr) =>
        logitsArray[curr] > logitsArray[maxIdx] ? curr : maxIdx,
      0
    );

    if (token === eos_id && generated_tokens.length === 0) {
      continue;
    }

    if (token === eos_id && generated_tokens.length > 0) {
      break;
    }

    generated_tokens.push(token);

    const nextTokenTensor = new ort.Tensor(
      "int32",
      new Int32Array([token]),
      [1, 1]
    );
    const nextTokenLenTensor = new ort.Tensor(
      "int32",
      new Int32Array([generated_tokens.length + 1]),
      [1]
    );

    const feeds = {
      [cachedDecode.inputNames[0]]: nextTokenTensor,
      [cachedDecode.inputNames[1]]: encoderOut,
      [cachedDecode.inputNames[2]]: nextTokenLenTensor,
    };

    let stateIdx = 3;
    for (const key of Object.keys(states)) {
      feeds[cachedDecode.inputNames[stateIdx]] = states[key];
      stateIdx++;
    }

    const cachedDecodeOutput = await cachedDecode.run(feeds);

    logits = cachedDecodeOutput["reversible_embedding"];
    states = {};
    for (const key of Object.keys(cachedDecodeOutput)) {
      if (key !== "reversible_embedding") {
        states[key] = cachedDecodeOutput[key];
      }
    }
  }

  // Use the tokenizer's decode method directly
  const text = await tokenizer.decode(generated_tokens, {
    skip_special_tokens: true,
    clean_up_tokenization_spaces: true,
  });

  return text;
}
