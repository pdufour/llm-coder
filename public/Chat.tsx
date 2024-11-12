import React, { useEffect, useRef, useState } from "react";
import { IFRAME_TEMPLATE } from "./constants/chat.ts";
import { useLLMHtmlGeneration } from "./hooks/useLLMHtmlGeneration.ts";

type Message = {
  role: "user" | "assistant";
  content: string;
};

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
