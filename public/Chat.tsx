// ChatInterface.tsx
import React, { useEffect, useRef, useState } from "react";
import { useLLMHtmlGeneration } from "./hooks/useLLMHtmlGeneration.ts";
import { CSPBuilder } from "./utils/csp.ts";

type Message = {
  role: "user" | "assistant";
  content: string;
};

const IFRAME_POSTMESSAGE_SCRIPT = /*js*/ `
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

const IFRAME_TAILWIND_CONFIG_SCRIPT = /*js*/ `;
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

const IFRAME_TAILWIND_URL = `https://cdn.tailwindcss.com`;

const { cspString: IFRAME_CSP, nonce: IFRAME_CSP_NONCE } =
  await new CSPBuilder()
    .addDirective("default-src", "'none'")
    .addDirective("img-src", "*")
    .addDirective("style-src", "'unsafe-inline'")
    .addDirective("script-src", "https://cdn.tailwindcss.com")
    .addScriptNonce() // Optionally add a nonce for inline scripts
    .build();

const IFRAME_TEMPLATE = `<!DOCTYPE html>
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

function ChatInterface() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const iframeRef = useRef<HTMLIFrameElement>(null);
  const { generateCode, isGenerating, error, generatedCode } =
    useLLMHtmlGeneration({
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
    });
  const [currentMessageId, setCurrentMessageId] = useState<number | null>(null);

  useEffect(() => {
    if (generatedCode) {
      // Send the content via postMessage
      iframeRef.current.contentWindow.postMessage(
        {
          type: "updateContent",
          content: generatedCode,
        },
        "*"
      );

      // console.log({ generatedCode });

      if (currentMessageId !== null) {
        setMessages((prev) =>
          prev.map((msg, idx) =>
            idx === currentMessageId ? { ...msg, content: generatedCode } : msg
          )
        );
      }
    }
  }, [generatedCode, currentMessageId]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;

    const userMessage: Message = {
      role: "user",
      content: input,
    };

    const assistantMessage: Message = {
      role: "assistant",
      content: "",
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setCurrentMessageId(messages.length + 1); // Index of assistant message
    setInput("");

    await generateCode(input);
  };

  // Function to convert error to string safely
  const stringifyError = (error: any): string => {
    if (!error) return "";
    return error.toString();
  };

  return (
    <div className="flex h-screen">
      {/* Chat Messages Section */}
      <div className="w-1/2 flex flex-col p-4 border-r">
        <div className="flex-1 overflow-auto space-y-4 mb-4">
          {messages.map((message, index) => {
            // Skip rendering assistant messages with empty content when not generating
            if (
              message.role === "assistant" &&
              !message.content &&
              !(index === currentMessageId && isGenerating)
            ) {
              return null;
            }

            return (
              <div
                key={index}
                className={`p-3 rounded-lg ${
                  message.role === "user"
                    ? "bg-blue-100 ml-auto max-w-[80%]"
                    : "bg-gray-100 mr-auto max-w-[80%]"
                }`}
              >
                {message.role === "assistant" ? (
                  <pre className="whitespace-pre-wrap font-mono text-sm overflow-x-auto">
                    {message.content ||
                      (index === currentMessageId &&
                        isGenerating &&
                        "Generating...")}
                  </pre>
                ) : (
                  <div>{message.content}</div>
                )}
              </div>
            );
          })}

          {/* Enhanced Error Handling */}
          {error && stringifyError(error) && (
            <div className="bg-red-100 text-red-700 p-3 rounded-lg mt-4">
              <strong>Error:</strong> {stringifyError(error)}
            </div>
          )}
        </div>

        {/* Input Form */}
        <form onSubmit={handleSubmit} className="flex gap-2">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Describe the HTML you want to generate..."
            className="flex-1 p-2 border rounded"
            disabled={isGenerating}
          />
          <button
            type="submit"
            disabled={isGenerating}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:opacity-50"
          >
            {isGenerating ? "Generating..." : "Send"}
          </button>
        </form>
      </div>

      {/* HTML Preview Section */}
      <div className="w-1/2 p-4 bg-gray-50">
        <iframe
          ref={iframeRef}
          className="w-full h-full border rounded-lg bg-white"
          title="HTML Preview"
          referrerPolicy="no-referrer"
          sandbox="allow-forms allow-scripts"
          srcDoc={IFRAME_TEMPLATE}
          csp={IFRAME_CSP}
          loading="lazy"
          allow=""
        />
      </div>
    </div>
  );
}

export const Chat = () => {
  return (
    <div className="h-screen">
      <ChatInterface />
    </div>
  );
};
