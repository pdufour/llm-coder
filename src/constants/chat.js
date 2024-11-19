import { CSPBuilder } from "../utils/csp.js";

export const LLM_VISION_MODEL_CONFIG = {
  modelConfig: {
    qwen2vl: {
      modelId: "pdufour/Qwen2-VL-2B-Instruct-ONNX-Q4-F16",
      generation: {
        baseModelId: "Qwen/Qwen2-VL-2B-Instruct",
        max_single_chat_length: 10,
        max_seq_length: 1024,
      },
    }
  },
};

export const LLM_HTML_MODEL_CONFIG = {
  modelConfig: {
    webllm: {
      modelId: "Qwen2.5-Coder-1.5B-Instruct-q4f16_1-MLC",
      generation: {
        max_tokens: 500,
        temperature: 0.3,
        top_p: 0.9,
      },
    },
    huggingface: {
      modelId: "Qwen/Qwen2.5-Coder-1.5B-Instruct",
      options: {
        device: "webgpu",
        dtype: "q4f16",
      },
      generation: {
        max_new_tokens: 500,
        temperature: 0.3,
        top_p: 0.9,
        do_sample: true,
      },
    },
  },
  backend: "webllm",
};

const { cspString: IFRAME_CSP, nonce: IFRAME_CSP_NONCE } =
  await new CSPBuilder()
    .addDirective("default-src", "'none'")
    .addDirective("img-src", "'none'")
    .addDirective("style-src", "'unsafe-inline'")
    .addDirective("script-src", "https://cdn.tailwindcss.com")
    .addScriptNonce()
    .build();

export const IFRAME_POSTMESSAGE_SCRIPT = /*js*/ `
window.addEventListener('message', function(event) {
  if (event.origin !== "${window.location.origin}") {
    return;
  }

  if (event.data.type === 'updateContent') {
    // Update the body content
    document.body.innerHTML = event.data.content;

    // Find and execute all script tags in the new content
    const scripts = document.body.querySelectorAll('script');
    scripts.forEach(script => {
      const newScript = document.createElement('script');
      if (script.src) {
        // If the script has a src attribute, copy it
        newScript.src = script.src;
      } else {
        // Otherwise, copy the inline script content
        newScript.textContent = script.textContent;
      }
      document.body.appendChild(newScript);
      // Remove the original script tag
      script.remove();
    });
  }
});
`;

export const IFRAME_TAILWIND_URL = `https://cdn.tailwindcss.com`;

export const IFRAME_TAILWIND_CONFIG_SCRIPT = /*js*/ `;
tailwind.config = {
  theme: {
    extend: {
      colors: {
        primary: "#3B82F6",
        secondary: "#10B981",
      },
    },
  },
};
`;

export const IFRAME_TEMPLATE = `<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="Content-Security-Policy" content="${IFRAME_CSP}">
    <script src="${IFRAME_TAILWIND_URL}" nonce="${IFRAME_CSP_NONCE}"></script>
    <style type="text/tailwindcss">
      @layer utilities {
        .content-auto {
          content-visibility: auto;
        }
      }
    </style>
    <script nonce="${IFRAME_CSP_NONCE}">
      ${IFRAME_TAILWIND_CONFIG_SCRIPT}
    </script>
</head>
<body class="bg-gray-50">
  <script nonce="${IFRAME_CSP_NONCE}">${IFRAME_POSTMESSAGE_SCRIPT}</script>
</body>
</html>`;
