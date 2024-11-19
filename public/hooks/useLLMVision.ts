import { AutoTokenizer, RawImage } from "@huggingface/transformers";
import { getModelJSON } from "@huggingface/transformers/utils/hub.js";
import { Tensor } from "@huggingface/transformers/utils/tensor.js";
import * as ort from "onnxruntime-web/webgpu";
import { logger } from "../utils/logging.ts";
import { float16ToInt64, int64ToFloat16 } from "../utils/math.ts";

async function logSessionIO(session: any, name: string) {
  if (!session) {
    logger.group(`[SESSION ${name}]`);
    logger.log("Session not loaded");
    logger.groupEnd();
    return;
  }
  logger.group(`[SESSION ${name}] Input details:`);
  session.inputNames.forEach((input: string) => {
    logger.group("Input:");
    logger.log(`    Name: ${input}`);
    logger.groupEnd();
  });
  logger.groupEnd();
}

const INPUT_IMAGE_SIZE = [960, 960] as const;
const HEIGHT_FACTOR = 10;
const WIDTH_FACTOR = 10;
const IMAGE_EMBED_SIZE = WIDTH_FACTOR * HEIGHT_FACTOR;
const MAX_SEQ_LENGTH = 1024;
const BASE_URL = "http://localhost:3004/onnx";
const BASE_MODEL = "Qwen/Qwen2-VL-2B-Instruct";
const QUANTIZATION = "q4f16";
const MAX_SINGLE_CHAT_LENGTH = 10;

export async function useLLMVision(
  imagePath: string,
  query: string,
  vision = true
) {
  logger.group("[CONFIG] Settings:");
  logger.log(`  INPUT_IMAGE_SIZE: ${JSON.stringify(INPUT_IMAGE_SIZE)}`);
  logger.log(`  HEIGHT_FACTOR: ${HEIGHT_FACTOR}`);
  logger.log(`  WIDTH_FACTOR: ${WIDTH_FACTOR}`);
  logger.log(`  MAX_SEQ_LENGTH: ${MAX_SEQ_LENGTH}`);
  logger.groupEnd();

  const suffix = QUANTIZATION ? `_${QUANTIZATION}` : "";

  logger.group("[SESSIONS] Loading and analyzing all ONNX sessions...");
  const startTime = performance.now();

  let ortSessionA, ortSessionB, ortSessionC, ortSessionD, ortSessionE;

  logger.groupEnd();

  logger.groupCollapsed("[MODEL] Loading configuration...");
  const config = (await getModelJSON(BASE_MODEL, "config.json")) as any;
  logger.log(`  num_hidden_layers: ${config.num_hidden_layers}`);
  logger.log(`  num_attention_heads: ${config.num_attention_heads}`);
  logger.log(`  num_key_value_heads: ${config.num_key_value_heads}`);
  logger.log(`  hidden_size: ${config.hidden_size}`);
  logger.groupEnd();

  const prompt_head_len = new Tensor("int64", new BigInt64Array([5n]), [1]);
  logger.tensor("prompt_head_len", prompt_head_len);

  let position_ids;
  let num_decode = 0;
  let history_len = new Tensor("int64", new BigInt64Array([0n]), [1]);
  logger.tensor("history_len", history_len);

  var pos_factor_v = BigInt(1 - IMAGE_EMBED_SIZE + WIDTH_FACTOR);

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
  logger.tensor("past_key_states", past_key_states);

  let past_value_states = past_key_states;
  logger.tensor("past_value_states", past_value_states);

  let attention_mask = new ort.Tensor(
    "float16",
    new Uint16Array([0xfbff]), // -65504.0 in float16
    [1]
  );
  logger.tensor("attention_mask", attention_mask);

  let pos_factor = new Tensor("float16", new Uint16Array([0]), [1]);
  logger.tensor("pos_factor", pos_factor);

  logger.groupCollapsed("[TOKENIZATION] Processing prompt...");
  const tokenizer = await AutoTokenizer.from_pretrained(BASE_MODEL);
  const prompt = `\n<|im_start|>user\n<|vision_start|><|vision_end|>${query}<|im_end|>\n<|im_start|>assistant\n`;
  const token = await tokenizer(prompt, {
    return_tensors: "pt",
    add_generation_prompt: false,
    tokenize: true,
  }).input_ids;
  logger.log("Token shape:", token.dims);
  logger.log("Token values:", Array.from(token.data));
  logger.groupEnd();

  const seq_length = token.dims[1];
  let ids_len = new Tensor("int64", new BigInt64Array([BigInt(seq_length)]), [
    1,
  ]);
  logger.tensor("ids_len", ids_len);

  let input_ids = new ort.Tensor(
    "int32",
    new Int32Array(MAX_SEQ_LENGTH).fill(0),
    [MAX_SEQ_LENGTH]
  );
  logger.tensor("input_ids (initial)", input_ids);

  input_ids.data.set(Array.from(token.data.slice(0, seq_length), Number));
  logger.tensor("input_ids (after set)", input_ids);

  const dummy = new ort.Tensor("int32", new Int32Array([0]), []);
  logger.tensor("dummy", dummy);

  logger.groupCollapsed("[INFERENCE] Running initial inference...");
  logger.log("Computing hidden states...");
  if (!ortSessionB) {
    ortSessionB = await ort.InferenceSession.create(
      `http://localhost:3004/onnx-dist/QwenVL_B${suffix}.onnx`,
      {
        executionProviders: ["webgpu"],
        logSeverityLevel: 0,
        logVerbosityLevel: 1,
        enableProfiling: true,
        enableCpuMemArena: true,
        graphOptimizationLevel: "all",
        executionMode: "sequential",
        intraOpNumThreads: 0,
        interOpNumThreads: 0,
      }
    );
  }
  let { hidden_states } = await ortSessionB.run({
    input_ids: input_ids,
    ids_len: ids_len,
  });

  logger.tensor("hidden_states (initial)", hidden_states);

  logger.groupCollapsed("[POSITION] Computing position IDs...");
  if (!ortSessionC) {
    ortSessionC = await ort.InferenceSession.create(
      `http://localhost:3004/onnx-dist/QwenVL_C${suffix}.onnx`,
      {
        executionProviders: ["webgpu"],
        logSeverityLevel: 2,
        logVerbosityLevel: 1,
        enableProfiling: true,
        enableCpuMemArena: true,
        graphOptimizationLevel: "all",
        executionMode: "sequential",
        intraOpNumThreads: 0,
        interOpNumThreads: 0,
      }
    );
  }
  ({ position_ids } = await ortSessionC.run({
    dummy: dummy,
  }));
  logger.tensor("position_ids (initial)", position_ids);
  logger.groupEnd();
  logger.groupEnd();

  // Process image
  if (vision) {
    logger.log("\n[IMAGE] Processing image...");
    let image = await RawImage.fromURL(imagePath);
    logger.log(`Original size: ${image.width}x${image.height}`);
    logger.log(`Original mode: ${image.mode}`);

    // image = await image.resize(INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]);
    console.log(`  Resized to: ${image.width}x${image.height}`);

    image = image.rgb();

    image = image.toTensor("CHW");
    image = image.to("float32");
    image = image.div_(255.0);
    const pixel_values = image.unsqueeze(0);
    logger.tensor("pixel_values", pixel_values);

    // Run session A for image embeddings
    if (!ortSessionA) {
      console.log("create session a");
      ortSessionA = await ort.InferenceSession.create(
        `http://localhost:3004/onnx-dist/QwenVL_A${suffix}.onnx`,
        {
          executionProviders: ["webgpu"],
          logSeverityLevel: 0,
          logVerbosityLevel: 0,
          enableProfiling: false,
          enableCpuMemArena: false,
          graphOptimizationLevel: "all",
          executionMode: "sequential",
          intraOpNumThreads: 0,
          interOpNumThreads: 0,
        }
      );
    }

    logger.log("session a run");
    const { image_embed } = await ortSessionA.run({
      pixel_values: pixel_values,
    });
    console.log("done session a");

    logger.tensor("image_embed", image_embed);

    ids_len = ids_len.add(BigInt(IMAGE_EMBED_SIZE));

    const split_factor = new Tensor(
      "int32",
      new Int32Array([
        MAX_SEQ_LENGTH - Number(ids_len.item()) - IMAGE_EMBED_SIZE,
      ]),
      [1]
    );

    const ids_len_minus = new Tensor(
      "int32",
      new Int32Array([Number(ids_len.item()) - Number(prompt_head_len.item())]),
      [1]
    );

    await ortSessionA.release();
    ortSessionA = null;

    logger.log("session d create");
    ortSessionD = await ort.InferenceSession.create(
      `http://localhost:3004/onnx-dist/QwenVL_D${suffix}.onnx`,
      {
        executionProviders: ["webgpu"],
        logSeverityLevel: 2,
        logVerbosityLevel: 0,
        enableProfiling: false,
        enableCpuMemArena: false,
        graphOptimizationLevel: "all",
        executionMode: "sequential",
        intraOpNumThreads: 0,
        interOpNumThreads: 0,
      }
    );

    logger.log("session d run");
    logger.tensor("image_embed", image_embed);
    logger.tensor("ids_len", ids_len);
    logger.tensor("ids_len_minus", ids_len_minus);
    logger.tensor("split_factor", split_factor);

    ({ hidden_states, position_ids } = await ortSessionD.run({
      "hidden_states.1": hidden_states,
      image_embed,
      ids_len,
      ids_len_minus,
      split_factor,
    }));

    logger.tensor("updated hidden_states", hidden_states);
    logger.tensor("updated position_ids", position_ids);

    await ortSessionD.release();
    ortSessionD = null;
  }

  logger.groupCollapsed("[GENERATION] Starting text generation...");
  const generationStartTime = performance.now();

  while (
    num_decode < MAX_SINGLE_CHAT_LENGTH &&
    Number(history_len.data[0]) < MAX_SEQ_LENGTH
  ) {
    let token_id;
    logger.groupCollapsed(`Step ${num_decode}`);
    logger.groupCollapsed("Session E inputs:");
    logger.tensor("hidden_states", hidden_states);
    logger.tensor("attention_mask", attention_mask);
    logger.tensor("past_key_states", past_key_states);
    logger.tensor("past_value_states", past_value_states);
    logger.tensor("history_len", history_len);
    logger.tensor("ids_len", ids_len);
    logger.tensor("position_ids", position_ids);
    logger.tensor("pos_factor", pos_factor);
    logger.groupEnd();

    if (!ortSessionE) {
      console.log("Create ortSessionE");
      ortSessionE = await ort.InferenceSession.create(
        `http://localhost:3004/onnx-dist/QwenVL_E_q4f16.onnx`,
        {
          executionProviders: ["wasm"],
          logSeverityLevel: 2,
          logVerbosityLevel: 0,
          enableProfiling: false,
          enableCpuMemArena: false,
          graphOptimizationLevel: "all",
          executionMode: "sequential",
          intraOpNumThreads: 0,
          interOpNumThreads: 0,
        }
      );
      console.log("outputNames", ortSessionE.outputNames);
    }

    // ort.env.debug = true;
    // ort.env.logLevel = "verbose";

    ({
      max_logit_ids: token_id,
      past_key_states: past_key_states,
      past_value_states: past_value_states,
    } = await ortSessionE.run({
      hidden_states,
      attention_mask,
      "past_key_states.1": past_key_states,
      "past_value_states.1": past_value_states,
      history_len,
      ids_len,
      position_ids,
      pos_factor,
    }));

    if (token_id === 151643 || token_id === 151645) {
      logger.log("Reached stop token");
      logger.groupEnd();
      break;
    }

    logger.tensor("New token_id", token_id);
    logger.log({ token_id });

    num_decode++;
    if (num_decode < 2) {
      logger.groupCollapsed("First decode step adjustments:");
      history_len = history_len.add(BigInt(ids_len.data[0]));
      logger.tensor("Updated history_len", history_len);

      ids_len = new ort.Tensor("int64", new BigInt64Array([1n]), [1]);
      logger.log(`Updated ids_len: ${ids_len.data[0]}`);

      attention_mask = new ort.Tensor("float16", new Uint16Array([0]), [1]);
      logger.log(`Updated attention_mask: ${attention_mask.data[0]}`);

      if (vision) {
        pos_factor = new Tensor(
          "float16",
          new Uint16Array([int64ToFloat16(pos_factor_v + ids_len.data[0])]),
          [1]
        );
      } else {
        pos_factor = new Tensor(
          "float16",
          new Uint16Array([int64ToFloat16(history_len.data[0] + BigInt(1))]),
          [1]
        );
      }

      logger.tensor("Updated pos_factor", pos_factor);
      logger.groupEnd();
    } else {
      logger.groupCollapsed(`Regular step ${num_decode} adjustments:`);
      history_len = history_len.add(BigInt(1));
      pos_factor = pos_factor.map((v) =>
        int64ToFloat16(float16ToInt64(v) + BigInt(1))
      );
      logger.tensor("Updated history_len", history_len);
      logger.tensor("Updated pos_factor", pos_factor);
      logger.groupEnd();
    }
    (input_ids.data as Int32Array)[0] = Number(token_id.data[0]);
    logger.tensor("Updated input_ids", input_ids);

    logger.groupCollapsed("Getting new hidden states...");
    const result_B = await ortSessionB.run({
      input_ids: input_ids,
      ids_len: ids_len,
    });
    hidden_states = result_B.hidden_states;
    logger.tensor("New hidden_states", hidden_states);
    logger.groupEnd();

    if (
      !Number.isInteger(token_id.data[0]) &&
      !["bigint", "number"].includes(typeof token_id.data[0])
    ) {
      throw new Error(`Token ID is not an integer`);
    } else {
      // Decode token
      const decoded = tokenizer.decode([...token_id.data]);
      logger.log(`Decoded token: ${decoded}`);
    }

    logger.groupEnd();
  }

  const generationTime = (performance.now() - generationStartTime) / 1000;
  logger.groupCollapsed("[PERFORMANCE] Generation complete:");
  logger.log(`  Total tokens generated: ${num_decode}`);
  logger.log(`  Generation time: ${generationTime.toFixed(2)}s`);
  logger.log(`  Speed: ${(num_decode / generationTime).toFixed(3)} tokens/s`);
  logger.groupEnd();
}
