import {
  AutoModel,
  AutoTokenizer,
  TextStreamer
} from "@huggingface/transformers";
import { CreateMLCEngine } from "@mlc-ai/web-llm";
import React from "react";

import { RawImage, Tensor } from "@huggingface/transformers";
import {
  getModelFile,
  getModelJSON,
} from "@huggingface/transformers/utils/hub.js";
import * as ort from "onnxruntime-web/webgpu";
import { logger } from "../utils/logging.js";
import { float16ToInt64, int64ToFloat16 } from "../utils/math.js";

export const SYSTEM_PROMPT = `You are a specialized HTML generator that creates complete, production-ready HTML using ONLY Tailwind CSS classes.`;

const INPUT_IMAGE_SIZE = [960, 960];
const HEIGHT_FACTOR = 10;
const WIDTH_FACTOR = 10;
const IMAGE_EMBED_SIZE = WIDTH_FACTOR * HEIGHT_FACTOR;
const QUANTIZATION = "q4f16";
const DEFAULT_SESSION_OPTIONS = {
  executionProviders: ["webgpu"],
  logSeverityLevel: 2,
  logVerbosityLevel: 1,
  // enableProfiling: true,
  // enableCpuMemArena: true,
  // graphOptimizationLevel: "all",
  // executionMode: "sequential",
  // intraOpNumThreads: 0,
  // interOpNumThreads: 0,
};

class Qwen2VLBackend {
  constructor(modelId, config) {
    this.modelId = modelId;
    this.config = config;
  }

  async generate(prompt, systemPrompt, callbacks, extras = {}) {
    const { imageURL } = extras;
    const vision = imageURL?.length || 0 > 0;
    logger.groupCollapsed("[CONFIG] Settings:");
    logger.log(`  INPUT_IMAGE_SIZE: ${JSON.stringify(INPUT_IMAGE_SIZE)}`);
    logger.log(`  HEIGHT_FACTOR: ${HEIGHT_FACTOR}`);
    logger.log(`  WIDTH_FACTOR: ${WIDTH_FACTOR}`);
    logger.log(`  max_seq_length: ${this.config.max_seq_length}`);
    logger.groupEnd();

    let ortSessionA, ortSessionB, ortSessionC, ortSessionD, ortSessionE;

    logger.groupCollapsed("[MODEL] Loading configuration...");
    const config = await getModelJSON(this.config.baseModelId, "config.json");
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
        this.config.max_seq_length *
        (config.hidden_size / config.num_attention_heads)
      ).fill(0),
      [
        config.num_hidden_layers,
        config.num_key_value_heads,
        this.config.max_seq_length,
        config.hidden_size / config.num_attention_heads,
      ]
    );
    logger.tensor("past_key_states", past_key_states);

    let past_value_states = past_key_states;
    logger.tensor("past_value_states", past_value_states);

    let attention_mask = new ort.Tensor(
      "float16", // -65504.0 in float16
      new Uint16Array([0xfbff]),
      [1]
    );
    logger.tensor("attention_mask", attention_mask);

    let pos_factor = new Tensor("float16", new Uint16Array([0]), [1]);
    logger.tensor("pos_factor", pos_factor);

    logger.groupCollapsed("[TOKENIZATION] Processing prompt...");
    const tokenizer = await AutoTokenizer.from_pretrained(this.config.baseModelId);
    const token_prompt = `\n<|im_start|>user\n<|vision_start|><|vision_end|>${prompt}<|im_end|>\n<|im_start|>assistant\n`;
    const token = await tokenizer(token_prompt, {
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
      new Int32Array(this.config.max_seq_length).fill(0),
      [this.config.max_seq_length]
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
        await getModelFile(this.modelId, `onnx/QwenVL_B_${QUANTIZATION}.onnx`),
        DEFAULT_SESSION_OPTIONS,
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
        await getModelFile(this.modelId, `onnx/QwenVL_C_${QUANTIZATION}.onnx`),
        DEFAULT_SESSION_OPTIONS,
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
      let image = await RawImage.fromURL(imageURL);
      logger.log(`Original size: ${image.width}x${image.height}`);

      image = await image.resize(INPUT_IMAGE_SIZE[0], INPUT_IMAGE_SIZE[1]);
      logger.log(`  Resized to: ${image.width}x${image.height}`);

      image = image.rgb();

      image = image.toTensor("CHW");
      image = image.to("float32");
      image = image.div_(255.0);
      const pixel_values = image.unsqueeze(0);
      logger.tensor("pixel_values", pixel_values);

      // Run session A for image embeddings
      if (!ortSessionA) {
        ortSessionA = await ort.InferenceSession.create(
          await getModelFile(this.modelId, `onnx/QwenVL_A_${QUANTIZATION}.onnx`),
          DEFAULT_SESSION_OPTIONS,
        );
      }

      logger.log("session a run");
      const { image_embed } = await ortSessionA.run({
        pixel_values: pixel_values,
      });
      logger.log("done session a");

      logger.tensor("image_embed", image_embed);

      ids_len = ids_len.add(BigInt(IMAGE_EMBED_SIZE));

      const split_factor = new Tensor(
        "int32",
        new Int32Array([
          this.config.max_seq_length - Number(ids_len.item()) - IMAGE_EMBED_SIZE,
        ]),
        [1]
      );

      const ids_len_minus = new Tensor(
        "int32",
        new Int32Array([
          Number(ids_len.item()) - Number(prompt_head_len.item()),
        ]),
        [1]
      );

      await ortSessionA.release();
      ortSessionA = null;

      logger.log("session d create");
      if (!ortSessionD) {
        ortSessionD = await ort.InferenceSession.create(
          await getModelFile(this.modelId, `onnx/QwenVL_D_${QUANTIZATION}.onnx`),
          DEFAULT_SESSION_OPTIONS,
        );
      }

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

    let accumulated = "";
    while (
      num_decode < this.config.max_single_chat_length &&
      Number(history_len.data[0]) < this.config.max_seq_length
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
        logger.log("Create ortSessionE");
        ortSessionE = await ort.InferenceSession.create(
          await getModelFile(this.modelId, `onnx/QwenVL_E_${QUANTIZATION}.onnx`),
          { ...DEFAULT_SESSION_OPTIONS, executionProviders: ["wasm"] },
        );
      }

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
      input_ids.data[0] = Number(token_id.data[0]);
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
        accumulated += decoded;
        callbacks.onToken(decoded);
        await scheduler?.scheduler?.yield();
      }

      logger.groupEnd();
    }

    const generationTime = (performance.now() - generationStartTime) / 1000;
    logger.groupCollapsed("[PERFORMANCE] Generation complete:");
    logger.log(`  Total tokens generated: ${num_decode}`);
    logger.log(`  Generation time: ${generationTime.toFixed(2)}s`);
    logger.log(`  Speed: ${(num_decode / generationTime).toFixed(3)} tokens/s`);
    logger.groupEnd();
    callbacks.onComplete(accumulated);
  }
}

class WebLLMBackend {
  constructor(modelId, config) {
    this.modelId = modelId
    this.config = config
  }

  async generate(prompt, systemPrompt, callbacks) {
    const engine = await CreateMLCEngine(this.modelId)

    const messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt }
    ]

    const asyncChunkGenerator = await engine.chat.completions.create({
      messages,
      stream: true,
      ...this.config
    })

    let accumulated = ""
    for await (const chunk of asyncChunkGenerator) {
      const newContent = chunk.choices[0]?.delta?.content || ""
      accumulated += newContent
      callbacks.onToken(newContent)
    }

    callbacks.onComplete(accumulated)
  }
}

class HuggingFaceBackend {
  constructor(modelId, options, config) {
    this.modelId = modelId
    this.options = options
    this.config = config
  }

  async generate(prompt, systemPrompt, callbacks) {
    const model = await AutoModel.from_pretrained(this.modelId, this.options)
    const tokenizer = await AutoTokenizer.from_pretrained(this.modelId)

    const messages = [
      { role: "system", content: systemPrompt },
      { role: "user", content: prompt }
    ]

    const text_inputs = tokenizer.apply_chat_template(messages, {
      tokenize: true,
      add_generation_prompt: true,
      return_dict: true,
      padding: true,
      truncation: true
    })

    let accumulated = ""
    const textStreamer = new TextStreamer(tokenizer, {
      skip_prompt: true,
      skip_special_tokens: true
    })

    textStreamer.on_finalized_text = text => {
      accumulated += text
      callbacks.onToken(text)
    }

    let isError = false

    try {
      await model.generate({
        ...text_inputs,
        ...this.config,
        streamer: textStreamer
      })
    } catch (error) {
      isError = true
      callbacks.onError(error);
      console.error(error);
    }

    if (!isError) {
      callbacks.onComplete(accumulated)
    }
  }
}

export function useLLMGeneration(
  modelConfig,
  systemPrompt,
  backend = "webllm"
) {
  const [isGenerating, setIsGenerating] = React.useState(false)
  const [error, setError] = React.useState(null)
  const [partialText, setPartialText] = React.useState("")
  const [fullText, setFullText] = React.useState("")

  const backendRef = React.useRef(null)

  React.useEffect(() => {
    if (backend === "qwen2vl" && modelConfig.qwen2vl) {
      backendRef.current = new Qwen2VLBackend(
        modelConfig.qwen2vl.modelId,
        modelConfig.qwen2vl.generation
      )
    }
    if (backend === "webllm" && modelConfig.webllm) {
      backendRef.current = new WebLLMBackend(
        modelConfig.webllm.modelId,
        modelConfig.webllm.generation
      )
    } else if (backend === "huggingface" && modelConfig.huggingface) {
      backendRef.current = new HuggingFaceBackend(
        modelConfig.huggingface.modelId,
        modelConfig.huggingface.options,
        modelConfig.huggingface.generation
      )
    }
  }, [backend, modelConfig])

  const generate = React.useCallback(
    async (prompt, extras) => {
      if (!backendRef.current) {
        throw new Error(`No backend configured for ${backend}`)
      }

      setIsGenerating(true)
      setError(null)
      setPartialText("")
      setFullText("")

      const callbacks = {
        onToken: token => {
          setPartialText(prev => prev + token)
        },
        onComplete: text => {
          setFullText(text)
          setIsGenerating(false)
        },
        onError: err => {
          setError(err)
          setIsGenerating(false)
        }
      }

      try {
        await backendRef.current.generate(prompt, systemPrompt, callbacks, extras)
      } catch (err) {
        callbacks.onError(err)
        logger.error(err);
      }
    },
    [backend, systemPrompt]
  )

  return {
    generate,
    isGenerating,
    error,
    partialText,
    fullText
  }
}
