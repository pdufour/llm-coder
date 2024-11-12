import {
  AutoModel,
  AutoProcessor,
  AutoTokenizer,
  env,
  RawImage,
} from "@huggingface/transformers";
import React from "react";
import { Home } from "./Home.tsx";

async function init() {
  const wasmPaths = new URL(
    ".",
    await import.meta.resolve("#onnxruntime-webgpu")
  ).toString();
  env.backends.onnx.wasm.wasmPaths = wasmPaths;

  // process_image();
}

async function process_image() {
  env.remoteHost = "http://localhost:3001";
  env.remotePathTemplate = "models/{model}";
  const url =
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/car.jpg";
  const image = await RawImage.fromURL(url);
  const model_id = "";

  const processor = await AutoProcessor.from_pretrained(model_id);

  const model = await AutoModel.from_pretrained(this.modelId, this.options);
  const tokenizer = await AutoTokenizer.from_pretrained(this.modelId);

  const messages = [
    { role: "system", content: "You are a" },
    { role: "user", content: prompt },
  ];

  const text_inputs = tokenizer.apply_chat_template(messages, {
    tokenize: true,
    add_generation_prompt: true,
    return_dict: true,
    padding: true,
    truncation: true,
  });

  // const res = await recognizeSpeechFromAudio(
  //   await read_audio(
  //     "https://huggingface.co/csukuangfj/sherpa-onnx-moonshine-base-en-int8/resolve/main/test_wavs/8k.wav",
  //     16000
  //   )
  // );
  // console.log({ res });
}

export function App() {
  return (
    <div className="min-h-screen min-w-full">
      <div className="mx-auto self-center flex flex-col items-center">
        <div className="flex flex-col w-full self-center items-center justify-center">
          <Home />
        </div>
      </div>
    </div>
  );
}

init();
