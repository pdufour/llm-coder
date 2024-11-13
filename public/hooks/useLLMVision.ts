import { AutoTokenizer, RawImage } from "@huggingface/transformers";
import { getModelJSON } from "@huggingface/transformers/utils/hub.js";
import { Tensor } from "@huggingface/transformers/utils/tensor.js";
import * as ort from "onnxruntime-web/webgpu";

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
        `${BASE_URL}/QwenVL_E${suffix}.onnx`,
        sessionOptions
      ),
    ]);

  // Load model configuration
  const config = (await getModelJSON(BASE_MODEL, "config.json")) as any;
  const prompt_head_len = new Tensor("int64", new BigInt64Array([5n]), [1]);

  // Initial variables
  let num_decode = 0;
  let history_len = new Tensor("int64", new BigInt64Array([0n]), [1]);

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
  let past_values_states = past_key_states;

  let attention_mask = new ort.Tensor("float16", new Uint16Array([-65504.0]), [
    1,
  ]); // -inf in float16
  let pos_factor = new ort.Tensor("float16", new Uint16Array([0]), [1]);

  // Process image
  let image = await RawImage.fromURL(imagePath);
  image = image.rgb();
  image = await image.resize(INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]);
  image = image.toTensor("CHW");
  image = image.to("float32");
  const pixel_values = image.unsqueeze(0);

  // Prepare messages
  const messages = [
    {
      role: "user",
      content: [
        { type: "text", text: query },
        { type: "image_url", image_url: "" },
      ],
    },
  ];

  // Tokenize input
  const tokenizer = await AutoTokenizer.from_pretrained(BASE_MODEL);
  const token = tokenizer.apply_chat_template(messages, {
    tokenize: true,
    return_tensors: "pt",
    add_generation_prompt: true,
  });

  // Prepare input tensors
  const seq_length = token.dims[1];
  let ids_len = new Tensor("int64", new BigInt64Array([BigInt(seq_length)]), [
    1,
  ]);
  let input_ids = new ort.Tensor(
    "int32",
    new Int32Array(MAX_SEQ_LENGTH).fill(0),
    [MAX_SEQ_LENGTH]
  );

  const tokenData = Array.from(token.data.slice(0, seq_length), Number);
  input_ids.data.set(tokenData);

  const dummy = new ort.Tensor(
    "int32",
    new Int32Array([0]),
    [] // scalar tensor
  );

  // Get hidden states from model B
  let { hidden_states } = await ortSessionB.run({
    input_ids: input_ids,
    ids_len: ids_len,
  });

  let { position_ids } = await ortSessionC.run({
    dummy: dummy,
  });

  // Process image with model A
  const { image_embed } = await ortSessionA.run({
    pixel_values: pixel_values,
  });

  // Calculate lengths
  const total_length = Number(ids_len.data[0]) + IMAGE_EMBED_SIZE;
  const updated_ids_len = BigInt(total_length);
  const remaining_length = Math.max(
    0,
    MAX_SEQ_LENGTH - total_length - IMAGE_EMBED_SIZE
  );

  // Prepare tensors for model D
  const split_factor = new Tensor("int32", new Int32Array([remaining_length]), [
    1,
  ]);
  const ids_len_minus = new Tensor(
    "int32",
    new Int32Array([Number(updated_ids_len - prompt_head_len.data[0])]),
    [1]
  );

  // Run model D with corrected shapes based on the actual dimensions
  ({ hidden_states, position_ids } = await ortSessionD.run({
    "hidden_states.1": new ort.Tensor(
      hidden_states.type,
      hidden_states.data,
      [1024, 1536] // Using the actual shape from hidden_states_shape
    ),
    image_embed: new ort.Tensor(
      image_embed.type,
      image_embed.data,
      [100, 1536] // Using the actual shape from image_embed_shape
    ),
    ids_len: new ort.Tensor("int64", new BigInt64Array([updated_ids_len]), [1]),
    ids_len_minus: ids_len_minus,
    split_factor: split_factor,
  }));
  console.log("Image process complete");
  console.log({ query });

  while (
    num_decode < MAX_SINGLE_CHAT_LENGTH &&
    Number(history_len.data[0]) < MAX_SEQ_LENGTH
  ) {
    // Run model E for next token prediction
    console.log("in", ortSessionE.inputNames);
    const result = await ortSessionE.run({
      hidden_states: hidden_states,
      attention_mask: attention_mask,
      "past_key_states.1": past_key_states,
      "past_value_states.1": past_values_states,
      history_len: history_len,
      ids_len: ids_len,
      position_ids: position_ids,
      pos_factor: pos_factor,
    });

    const token_id = Number(result.token_id.data[0]);
    past_key_states = result.past_key_states;
    past_values_states = result.past_values_states;

    // Check for stop tokens
    if (token_id === 151643 || token_id === 151645) {
      break;
    }

    num_decode++;

    if (num_decode < 2) {
      // Update history length
      const newHistoryLen =
        Number(history_len.data[0]) + Number(ids_len.data[0]);
      history_len = new ort.Tensor(
        "int64",
        new BigInt64Array([BigInt(newHistoryLen)]),
        [1]
      );

      // Update ids_len
      ids_len = new ort.Tensor("int64", new BigInt64Array([1n]), [1]);

      // Update attention mask
      attention_mask = new ort.Tensor("float16", new Uint16Array([0]), [1]);

      // Update pos_factor
      var pos_factor_v = 1 - IMAGE_EMBED_SIZE + WIDTH_FACTOR;
      const newPosFactor = pos_factor_v + Number(ids_len.data[0]);
      pos_factor = new ort.Tensor("float16", new Uint16Array([newPosFactor]), [
        1,
      ]);
    } else {
      // Increment history length
      const newHistoryLen = Number(history_len.data[0]) + 1;
      history_len = new ort.Tensor(
        "int64",
        new BigInt64Array([BigInt(newHistoryLen)]),
        [1]
      );

      // Increment pos_factor
      const newPosFactor = Number(pos_factor.data[0]) + 1;
      pos_factor = new ort.Tensor("float16", new Uint16Array([newPosFactor]), [
        1,
      ]);
    }

    // Update input_ids with new token
    input_ids = new ort.Tensor("int32", new Int32Array([token_id]), [1]);

    // Get new hidden states
    const result_B = await ortSessionB.run({
      input_ids: input_ids,
      ids_len: ids_len,
    });
    hidden_states = result_B.hidden_states;

    // Decode and print token
    const decoded = await tokenizer.decode(new Int32Array([token_id]));
    console.log({ decoded });
  }
}
