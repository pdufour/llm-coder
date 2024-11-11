import { env } from "@huggingface/transformers";
import React from "react";
import { Home } from "./Home.tsx";

async function init() {
  const wasmPaths = new URL(
    ".",
    await import.meta.resolve("#onnxruntime-webgpu")
  ).toString();

  env.backends.onnx.wasm.wasmPaths = wasmPaths;

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
