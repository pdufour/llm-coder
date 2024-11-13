import { AutoTokenizer } from "@huggingface/transformers";
import { getModelJSON } from "@huggingface/transformers/utils/hub.js";
import { Tensor } from "@huggingface/transformers/utils/tensor.js";
import * as ort from "onnxruntime-web/webgpu";

function toFloat16(num) {
  // Create a new DataView of an ArrayBuffer with 4 bytes (32 bits)
  const buffer = new ArrayBuffer(4);
  const dataView = new DataView(buffer);

  // Set the 32-bit float value in the DataView
  dataView.setFloat32(0, num, true); // true for little-endian

  // Extract the 16-bit unsigned integer representation
  const float16Bits = dataView.getUint16(0, true);

  // Return the original value and the float16 representation
  return [num, float16Bits];
}

// Helper functions for logging
function logTensor(name: string, tensor: any) {
  console.log(`\n[NUMPY] ${name}:`);
  console.log(`  Shape: ${JSON.stringify(tensor.dims || tensor.shape)}`);
  console.log(`  Type: ${tensor.type}`);
  if (tensor.data) {
    const data = Array.from(tensor.data).map((val) =>
      typeof val === "bigint" ? Number(val) : val
    );
    try {
      const min = Math.min(...data);
      const max = Math.max(...data);
      console.log(`  Min/Max: ${min.toFixed(6)} / ${max.toFixed(6)}`);
    } catch (e) {
      console.log(`  Min/Max: Unable to calculate for this data type`);
    }

    // For displaying values, convert BigInts to strings to prevent errors
    const displayData = Array.from(tensor.data).map((val) =>
      typeof val === "bigint" ? val.toString() : val
    );
    console.log(`  First few values: [${displayData.slice(0, 5).join(", ")}]`);
    console.log(`  Last few values: [${displayData.slice(-5).join(", ")}]`);
  }
}

// Use it in our logging function
async function logSessionIO(session: any, name: string) {
  // Original basic logging
  console.log(`\n[SESSION ${name}] Input details:`);
  session.inputNames.forEach((input: string) => {
    console.log(`  Input:`);
    console.log(`    Name: ${input}`);
  });
}

// Constants and Configuration
const INPUT_IMAGE_SIZE = [960, 960] as const;
const HEIGHT_FACTOR = 10;
const WIDTH_FACTOR = 10;
const IMAGE_EMBED_SIZE = WIDTH_FACTOR * HEIGHT_FACTOR;
const MAX_SEQ_LENGTH = 1024;
const BASE_URL = "http://localhost:3004/onnx";
const BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct";
const QUANTIZATION = "q4f16";
const MAX_SINGLE_CHAT_LENGTH = 341;

export async function useLLMVision(imagePath: string, query: string) {
  console.log("\n[STARTUP] Beginning QwenVL initialization...");

  console.log("\n[CONFIG] Settings:");
  console.log(`  INPUT_IMAGE_SIZE: ${JSON.stringify(INPUT_IMAGE_SIZE)}`);
  console.log(`  HEIGHT_FACTOR: ${HEIGHT_FACTOR}`);
  console.log(`  WIDTH_FACTOR: ${WIDTH_FACTOR}`);
  console.log(`  MAX_SEQ_LENGTH: ${MAX_SEQ_LENGTH}`);

  const suffix = QUANTIZATION ? `_${QUANTIZATION}` : "";
  const sessionOptions: ort.InferenceSession.SessionOptions = {
    executionProviders: ["webgpu"],
    logSeverityLevel: 0,
    logVerbosityLevel: 0,
    enableProfiling: true,
    enableCpuMemArena: true,
    graphOptimizationLevel: "all",
    executionMode: "sequential",
  };

  console.log("\n[SESSIONS] Loading and analyzing all ONNX sessions...");
  const startTime = performance.now();

  // Load all ONNX sessions in parallel
  const [ortSessionA, ortSessionB, ortSessionC, ortSessionD, ortSessionE] =
    await Promise.all([
      ort.InferenceSession.create(
        `${BASE_URL}/QwenVL_A${suffix}.onnx`,
        sessionOptions
      ),
      ort.InferenceSession.create(
        `${BASE_URL}/QwenVL_B${suffix}.onnx`,
        sessionOptions
      ),
      ort.InferenceSession.create(
        `${BASE_URL}/QwenVL_C${suffix}.onnx`,
        sessionOptions
      ),
      ort.InferenceSession.create(
        `${BASE_URL}/QwenVL_D${suffix}.onnx`,
        sessionOptions
      ),
      ort.InferenceSession.create(
        `${BASE_URL}/QwenVL_E_uint8.onnx`,
        sessionOptions
      ),
    ]);

  for (
    let i = 0;
    i <
    [ortSessionA, ortSessionB, ortSessionC, ortSessionD, ortSessionE].length;
    i++
  ) {
    const session = [
      ortSessionA,
      ortSessionB,
      ortSessionC,
      ortSessionD,
      ortSessionE,
    ][i];
    const name = String.fromCharCode(65 + i);
    await logSessionIO(session, name);
  }

  // Load model configuration
  console.log("\n[MODEL] Loading configuration...");
  const config = (await getModelJSON(BASE_MODEL, "config.json")) as any;
  console.log("\n[MODEL] Configuration:");
  console.log(`  num_hidden_layers: ${config.num_hidden_layers}`);
  console.log(`  num_attention_heads: ${config.num_attention_heads}`);
  console.log(`  num_key_value_heads: ${config.num_key_value_heads}`);
  console.log(`  hidden_size: ${config.hidden_size}`);

  const prompt_head_len = new Tensor("int64", new BigInt64Array([5n]), [1]);
  logTensor("prompt_head_len", prompt_head_len);

  // Initial variables
  let num_decode = 0;
  let history_len = new Tensor("int64", new BigInt64Array([0n]), [1]);
  logTensor("history_len", history_len);

  var pos_factor_v = 1 - IMAGE_EMBED_SIZE + WIDTH_FACTOR;
  console.log("pos_factor_v: ", pos_factor_v);

  let past_key_states = new ort.Tensor(
    "float16",
    new Uint16Array(
      config.num_hidden_layers *
        config.num_key_value_heads *
        MAX_SEQ_LENGTH *
        (config.hidden_size / config.num_attention_heads)
    ).fill(0),
    [
      config.num_hidden_layers,
      config.num_key_value_heads,
      MAX_SEQ_LENGTH,
      config.hidden_size / config.num_attention_heads,
    ]
  );
  logTensor("past_key_states", past_key_states);

  let past_value_states = past_key_states;
  logTensor("past_value_states", past_value_states);

  // const attentionMaskVal = new Uint16Array(1);
  // attentionMaskVal[0] = 0xfc00;

  let attention_mask = new ort.Tensor("float16", new Uint16Array([-65504.0]), [
    1,
  ]);
  // let attention_mask = new ort.Tensor("float16", attentionMaskVal, [1]);

  logTensor("attention_mask", attention_mask);

  let pos_factor = new ort.Tensor("float16", new Uint16Array([0]), [1]);
  logTensor("pos_factor", pos_factor);

  // Process image
  // VISION
  // console.log("\n[IMAGE] Processing image...");
  // const imageStartTime = performance.now();
  // let image = await RawImage.fromURL(imagePath);
  // console.log(`  Original size: ${image.width}x${image.height}`);
  // console.log(`  Original mode: ${image.mode}`);

  // image = image.rgb();
  // image = await image.resize(INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]);
  // console.log(`  Resized to: ${image.width}x${image.height}`);

  // image = image.toTensor("CHW");
  // image = image.to("float32");
  // const pixel_values = image.unsqueeze(0);
  // logTensor("pixel_values", pixel_values);
  // END VISION

  // Prepare messages
  const messages = [
    {
      role: "user",
      content: [
        { type: "text", text: query },
        // { type: "image_url", image_url: "" },
      ],
    },
  ];

  // Tokenize input
  console.log("\n[TOKENIZATION] Processing prompt...");
  const tokenizer = await AutoTokenizer.from_pretrained(BASE_MODEL);
  const token = tokenizer.apply_chat_template(messages, {
    tokenize: true,
    return_tensors: "pt",
    add_generation_prompt: true,
  });
  console.log("Token shape:", token.dims);
  console.log("Token values:", Array.from(token.data));

  // Prepare input tensors
  const seq_length = token.dims[1];
  let ids_len = new Tensor("int64", new BigInt64Array([BigInt(seq_length)]), [
    1,
  ]);
  logTensor("ids_len", ids_len);

  let input_ids = new ort.Tensor(
    "int32",
    new Int32Array(MAX_SEQ_LENGTH).fill(0),
    [MAX_SEQ_LENGTH]
  );
  logTensor("input_ids (initial)", input_ids);

  const tokenData = Array.from(token.data.slice(0, seq_length), Number);
  input_ids.data.set(tokenData);
  logTensor("input_ids (after set)", input_ids);

  const dummy = new ort.Tensor("int32", new Int32Array([0]), []);
  logTensor("dummy", dummy);

  // Get hidden states from model B
  console.log("\n[INFERENCE] Running initial inference...");
  console.log("Computing hidden states...");
  let { hidden_states } = await ortSessionB.run({
    input_ids: input_ids,
    ids_len: ids_len,
  });
  logTensor("hidden_states (initial)", hidden_states);

  let { position_ids } = await ortSessionC.run({
    dummy: dummy,
  });
  logTensor("position_ids (initial)", position_ids);

  // VISION
  // Process image with model A
  // console.log("\n[VISION] Processing vision inputs...");
  // const { image_embed } = await ortSessionA.run({
  //   pixel_values: pixel_values,
  // });
  // logTensor("image_embed", image_embed);

  // // Calculate lengths
  // const total_length = Number(ids_len.data[0]) + IMAGE_EMBED_SIZE;
  // const updated_ids_len = BigInt(total_length);
  // const remaining_length = Math.max(
  //   0,
  //   MAX_SEQ_LENGTH - total_length - IMAGE_EMBED_SIZE
  // );

  // // Prepare tensors for model D
  // const split_factor = new Tensor("int32", new Int32Array([remaining_length]), [
  //   1,
  // ]);
  // logTensor("split_factor", split_factor);

  // const ids_len_minus = new Tensor(
  //   "int32",
  //   new Int32Array([Number(updated_ids_len - prompt_head_len.data[0])]),
  //   [1]
  // );
  // logTensor("ids_len_minus", ids_len_minus);

  // console.log("\nProcessing vision embeddings...");
  // console.log("Session D inputs:");
  // console.log(`  hidden_states shape: ${hidden_states.dims}`);
  // console.log(`  image_embed shape: ${image_embed.dims}`);
  // console.log(`  ids_len: ${updated_ids_len}`);
  // console.log(`  ids_len_minus: ${ids_len_minus.data[0]}`);
  // console.log(`  split_factor: ${split_factor.data[0]}`);

  // ({ hidden_states, position_ids } = await ortSessionD.run({
  //   "hidden_states.1": new ort.Tensor(
  //     hidden_states.type,
  //     hidden_states.data,
  //     [1024, 1536]
  //   ),
  //   image_embed: new ort.Tensor(
  //     image_embed.type,
  //     image_embed.data,
  //     [100, 1536]
  //   ),
  //   ids_len: new ort.Tensor("int64", new BigInt64Array([updated_ids_len]), [1]),
  //   ids_len_minus: ids_len_minus,
  //   split_factor: split_factor,
  // }));

  // console.log("\nImage process complete");
  // console.log(
  //   `Time taken: ${((performance.now() - imageStartTime) / 1000).toFixed(2)}s`
  // );
  // END VISION
  // console.log({ query });

  console.log("\n[GENERATION] Starting text generation...");
  const generationStartTime = performance.now();

  while (
    num_decode < MAX_SINGLE_CHAT_LENGTH &&
    Number(history_len.data[0]) < MAX_SEQ_LENGTH
  ) {
    let token_id: number;

    console.log(`\n[GENERATION] Step ${num_decode}`);
    console.log("Session E inputs:");
    console.log(`  hidden_states shape: ${hidden_states.dims}`);
    console.log(`  attention_mask shape: ${attention_mask.dims}`);
    console.log(`  past_key_states shape: ${past_key_states.dims}`);
    console.log(`  past_value_states shape: ${past_value_states.dims}`);
    console.log(`  history_len shape: ${history_len.dims}`);
    console.log(`  ids_len shape: ${ids_len.dims}`);
    console.log(`  position_ids shape: ${position_ids.dims}`);
    console.log(`  pos_factor shape: ${pos_factor.dims}`);

    ({
      max_logit_ids: token_id,
      past_key_states,
      past_value_states,
    } = await ortSessionE.run({
      hidden_states: hidden_states,
      attention_mask: attention_mask,
      "past_key_states.1": past_key_states,
      "past_value_states.1": past_value_states,
      history_len: history_len,
      ids_len: ids_len,
      position_ids: position_ids,
      pos_factor: pos_factor,
    }));
    if (token_id === 151643 || token_id === 151645) {
      console.log("\n[GENERATION] Reached stop token");
      break;
    }

    logTensor("New token_id", token_id);
    console.log({ token_id });

    num_decode++;
    if (num_decode < 2) {
      console.log("\n[GENERATION] First decode step adjustments:");
      // const newHistoryLen =
      //   Number(history_len.data[0]) + Number(ids_len.data[0]);
      // history_len = new ort.Tensor(
      //   "int64",
      //   new BigInt64Array([BigInt(newHistoryLen)]),
      //   [1]
      // );
      history_len.add(BigInt(ids_len.data[0]));

      logTensor("Updated history_len", history_len);

      ids_len = new ort.Tensor("int64", new BigInt64Array([1n]), [1]);
      console.log(`  Updated ids_len: ${ids_len.data[0]}`);

      attention_mask = new ort.Tensor("float16", new Uint16Array([0]), [1]);
      console.log(`  Updated attention_mask: ${attention_mask.data[0]}`);

      // VISION
      // const newPosFactor = pos_factor_v + Number(ids_len.data[0]);
      // pos_factor = new ort.Tensor("float16", new Uint16Array([newPosFactor]), [
      //   1,
      // ]);
      // END VISION

      // NON_VISION
      pos_factor = new ort.Tensor(
        "float16",
        new Uint16Array([Number(history_len.data[0]) + 1]),
        [1]
      ); // Shape as (1,)
      // END NON VISION
      console.log(`  Updated pos_factor: ${pos_factor.data[0]}`);
    } else {
      console.log(`\n[GENERATION] Regular step ${num_decode} adjustments:`);
      // const newHistoryLen = Number(history_len.data[0]) + 1;
      // history_len = new ort.Tensor(
      //   "int64",
      //   new BigInt64Array([BigInt(newHistoryLen)]),
      //   [1]
      // );
      // console.log(`  Updated history_len: ${newHistoryLen}`);
      history_len.add(BigInt(1));

      // const newPosFactor = Number(pos_factor.data[0]) + 1;
      // pos_factor = new ort.Tensor("float16", new Uint16Array([newPosFactor]), [
      //   1,
      // ]);
      pos_factor.add(BigInt(1));

      logTensor("Updated pos_factor", pos_factor);
      // console.log(`  Updated pos_factor: ${newPosFactor}`);
    }

    input_ids.data.set(tokenData, 0);
    logTensor("  Updated input_ids", input_ids);

    console.log("\nGetting new hidden states...");
    const result_B = await ortSessionB.run({
      input_ids: input_ids,
      ids_len: ids_len,
    });
    hidden_states = result_B.hidden_states;
    logTensor("New hidden_states", hidden_states);

    if (!Number.isInteger(token_id)) {
      console.error(`Token ID is not an integer`);
    } else {
      const decoded = await tokenizer.decode(new Int32Array([token_id]));
      console.log(`Decoded token: ${decoded}`);
    }
  }

  const generationTime = (performance.now() - generationStartTime) / 1000;
  console.log(`\n\n[PERFORMANCE] Generation complete:`);
  console.log(`  Total tokens generated: ${num_decode}`);
  console.log(`  Generation time: ${generationTime.toFixed(2)}s`);
  console.log(`  Speed: ${(num_decode / generationTime).toFixed(3)} tokens/s`);
}
