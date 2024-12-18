import { env } from "@huggingface/transformers";
import React from "react";
import { Home } from "./Home.js";

const h = React.createElement;

async function init() {
  const wasmPaths = new URL(
    ".",
    await import.meta.resolve("#onnxruntime-webgpu")
  ).toString();
  env.backends.onnx.wasm.wasmPaths = wasmPaths;
}

export const App = () => {
  return h(
    "div",
    { className: "min-h-screen min-w-full" },
    h(
      "div",
      { className: "mx-auto self-center flex flex-col items-center" },
      h(
        "div",
        {
          className:
            "flex flex-col w-full self-center items-center justify-center",
        },
        h(Home)
      )
    )
  );
}

init();
