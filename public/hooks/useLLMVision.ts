import { AutoTokenizer, RawImage } from "@huggingface/transformers";
import { getModelJSON } from "@huggingface/transformers/utils/hub.js";
import { Tensor } from "@huggingface/transformers/utils/tensor.js";
import * as ort from "onnxruntime-web/webgpu";

const INPUT_IMAGE_SIZE = [960, 960];
const HEIGHT_FACTOR = 10;
const WIDTH_FACTOR = 10;
const IMAGE_RESIZE = [HEIGHT_FACTOR * 28, WIDTH_FACTOR * 28];
const MAX_SEQ_LENGTH = 1024;
const MAX_SINGLE_CHAT_LENGTH = 341;
const IMAGE_EMBED_SIZE = WIDTH_FACTOR * HEIGHT_FACTOR;

const BASE_URL = "http://localhost:3004/onnx";
const BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct";
const QUANTIZATION = "q4f16";

export async function useLLMVision(imagePath, query) {
  // suffix is _ + quant if quant is set
  const suffix = QUANTIZATION ? `_${QUANTIZATION}` : "";
  const sessionOptions = {
    executionProviders: ["webgpu"],
    logSeverityLevel: 2,
    logVerbosityLevel: 2,
    enableProfiling: true,
    enableCpuMemArena: true,
    graphOptimizationLevel: "all",
    executionMode: "sequential",
  };

  console.log("Loading session a");
  const ortSessionA = await ort.InferenceSession.create(
    `${BASE_URL}/QwenVL_A${suffix}.onnx`,
    {
      executionProviders: ["webgpu"], // Switch to ["cpu"] if WebGPU causes issues
      logSeverityLevel: 2, // Verbose logging for detailed errors,
      logVerbosityLevel: 2, // Add this for more verbose logging
      enableProfiling: true, // Add this to get performance data
      enableCpuMemArena: true, // Add this for memory usage info
      graphOptimizationLevel: "all", // Add this to see optimization steps
      executionMode: "sequential", // Add this to track execution flow
    }
  );
  console.log("Loading session b");
  const ortSessionB = await ort.InferenceSession.create(
    `${BASE_URL}/QwenVL_B${suffix}.onnx`,
    sessionOptions
  );
  console.log("Loading session c");

  const ortSessionC = await ort.InferenceSession.create(
    `${BASE_URL}/QwenVL_C${suffix}.onnx`,
    sessionOptions
  );
  console.log("Loading session d");

  const ortSessionD = await ort.InferenceSession.create(
    `${BASE_URL}/QwenVL_D${suffix}.onnx`,
    sessionOptions
  );
  console.log("Loading session e");

  const E = await ort.InferenceSession.create(
    `${BASE_URL}/QwenVL_E${suffix}.onnx`,
    sessionOptions
  );
  console.log("Got sessions");

  // Read config
  const config = await getModelJSON(BASE_MODEL, "config.json");
  // console.log({ config });

  // constants
  const prompt_head_len = new Tensor("int64", new BigInt64Array([5n]), [1]);

  // num_heads = model.config.num_attention_heads
  // num_key_value_heads = model.config.num_key_value_heads
  // head_dim = model.config.hidden_size // num_heads
  // num_layers = model.config.num_hidden_layers
  // hidden_size = model.config.hidden_size

  // const prompt = `\n<|im_start|>user\n<|vision_start|><|vision_end|>${query}<|im_end|>\n<|im_start|>assistant\n`;
  // GET IMAGE DATA

  // const url =
  //   "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg";
  // const image = await RawImage.fromURL(url);

  let image = await RawImage.fromURL(
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg"
  );

  image = image.rgb();
  image = await image.resize(INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]);
  image = image.toTensor("CHW");
  image = image.to("float32");

  const pixel_values = image.unsqueeze(0);

  // EMBED TOKEN
  const messages = [
    {
      role: "user",
      content: [
        { type: "text", text: query },
        { type: "image_url", image_url: "" }, // Empty image for <|vision_start|><|vision_end|>
      ],
    },
  ];

  // py: prompt = f"\n<|im_start|>user\n<|vision_start|><|vision_end|>{query}<|im_end|>\n<|im_start|>assistant\n"
  // js:  "<|im_start|>system
  // You are a helpful assistant.<|im_end|>
  // <|im_start|>user
  // Describe this image<|vision_start|><|image_pad|><|vision_end|><|im_end|>
  // <|im_start|>assistant
  // "

  const tokenizer = await AutoTokenizer.from_pretrained(BASE_MODEL);
  const token = tokenizer.apply_chat_template(messages, {
    tokenize: true,
    return_tensors: "pt",
    add_generation_prompt: true,
    // return_dict: true,
  });

  const ids_len = new Tensor(
    "int64",
    new BigInt64Array([BigInt(token.dims[1])]),
    [1]
  );

  const input_ids = new ort.Tensor(
    "int32",
    new Int32Array(MAX_SEQ_LENGTH).fill(0),
    [MAX_SEQ_LENGTH]
  );

  const tokenData = Array.from(token.data.slice(0, token.dims[1]), (bigInt) =>
    Number(bigInt)
  );
  input_ids.data.set(tokenData);

  const history_len = new BigInt64Array([0n]);

  const past_key_states = new ort.Tensor(
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
  const past_values_states = past_key_states;
  const attention_mask = new ort.Tensor(
    "float16",
    new Uint16Array([-65504.0]),
    [1]
  );
  const pos_factor = new ort.Tensor(
    "float16",
    new Uint16Array([0.0]),
    [] // empty array for scalar tensor, not [0]
  );

  // export var pos_factor_v = 1 - image_embed_size + WIDTH_FACTOR;
  const pos_factor_v = 1 - IMAGE_EMBED_SIZE + WIDTH_FACTOR;

  // export var dummy = np.array(0, __kwargtrans__({ dtype: np.int32 }));
  const dummy = new ort.Tensor(
    "int32",
    new Int32Array([0]),
    [] // scalar tensor
  );

  // export var num_decode = 0;
  const num_decode = 0;

  // get hidden states
  // export var hidden_states = ort_session_B.run(
  //     [out_name_B0],
  //     dict([[in_name_B0, input_ids], [in_name_B1, ids_len]]))[0];
  // console.log(ortSessionB.inputNames);
  const hidden_states = (
    await ortSessionB.run({
      input_ids: input_ids,
      ids_len: ids_len,
    })
  ).hidden_states;

  // get position ids
  // var position_ids = ort_session_C.run(
  //   [out_name_C0],
  //   dict([[in_name_C0, dummy]])
  // );
  const position_ids = await ortSessionC.run({
    dummy: dummy,
  });

  console.log({ position_ids });

  console.log("\nStart to Process the Image..."); // print("\nStart to Process the Image...");
  const start_time = Date.now(); // var start_time = time.time();

  // var image_embed = ort_session_A.run([out_name_A0], dict([[in_name_A0, pixel_values]]))[0];
  const image_embed = (
    await ortSessionA.run({
      pixel_values: pixel_values,
    })
  ).image_embed;

  // ids_len += image_embed_size;

  ids_len.add(BigInt(IMAGE_EMBED_SIZE));
  // console.log({ ids_len });

  // var split_factor = np.array(max_seq_len - ids_len[0] - image_embed_size, __kwargtrans__({ dtype: np.int32 }));
  const split_factor = new Tensor(
    "int32",
    new Int32Array([
      Number(
        BigInt(MAX_SEQ_LENGTH) - ids_len.data[0] - BigInt(IMAGE_EMBED_SIZE)
      ),
    ]),
    []
  );

  // var ids_len_minus = np.array(ids_len[0] - prompt_head_len[0], __kwargtrans__({ dtype: np.int32 }));
  const ids_len_minus = new Tensor(
    "int32",
    new Int32Array([Number(ids_len.data[0] - prompt_head_len.data[0])]),
    []
  );

  console.log(ortSessionD.inputNames);
  // // var __left0__ = ort_session_D.run(
  // //     [out_name_D0, out_name_D1],
  // //     dict([[in_name_D0, hidden_states], [in_name_D1, image_embed],
  // //          [in_name_D2, ids_len], [in_name_D3, ids_len_minus], [in_name_D4, split_factor]])
  // // );
  const res = await ortSessionD.run({
    "hidden_states.1": hidden_states,
    image_embed: image_embed,
    ids_len: ids_len,
    ids_len_minus: ids_len_minus,
    split_factor: split_factor,
  });
  console.log({ res });

  // // var hidden_states = __left0__[0];
  // // var position_ids = __left0__[1];
  // hidden_states = new_hidden_states;
  // position_ids = new_position_ids;

  // const end_time = Date.now(); // var end_time = time.time();
  // // print("\nImage Process Complete. Time Cost: {}".format(end_time - start_time))
  // console.log(
  //   `\nImage Process Complete. Time Cost: ${(end_time - start_time) / 1000}s`
  // );

  if (1 == 1) {
    return;
  }

  return response;
}
