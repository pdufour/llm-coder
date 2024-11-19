import { Code, MinimizeIcon, Send } from "lucide-react";
import React, { useEffect, useRef, useState } from "react";
import { IFRAME_TEMPLATE, LLM_HTML_MODEL_CONFIG } from "./constants/chat.js";
import { useLLMHtmlGeneration } from "./hooks/useLLMHtmlGeneration.js";

const h = React.createElement;

export function Chat() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isCodePanelOpen, setIsCodePanelOpen] = useState(false);
  const iframeRef = useRef(null);
  const [buildStage, setBuildStage] = useState(0);

  const { generateCode, isGenerating, error, generatedCode } =
    useLLMHtmlGeneration(LLM_HTML_MODEL_CONFIG);
  const [currentMessageId, setCurrentMessageId] = useState(null);

  useEffect(() => {
    if (generatedCode) {
      iframeRef.current?.contentWindow?.postMessage(
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

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isGenerating) return;

    const userMessage = { role: "user", content: input };
    const assistantMessage = { role: "assistant", content: "" };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setCurrentMessageId(messages.length + 1);
    setInput("");
    setIsCodePanelOpen(true);

    await generateCode(input);
  };

  return h(
    "div",
    { className: "min-h-screen bg-gray-950 text-gray-200" },
    !isGenerating && !messages.length
      ? h(
        "div",
        { className: "absolute inset-0 flex items-center justify-center" },
        h(
          "div",
          {
            className:
              "text-5xl font-bold tracking-tighter text-gray-800/20 select-none",
          },
          "Prompt. Create. Deploy."
        )
      )
      : h(
        "div",
        { className: "absolute inset-0 animate-fade-in" },
        h(
          "div",
          {
            className:
              "h-screen w-full bg-white rounded-none overflow-hidden shadow-2xl",
          },
          h("iframe", {
            ref: iframeRef,
            className: "w-full h-full border-0",
            title: "HTML Preview",
            referrerPolicy: "no-referrer",
            sandbox: "allow-forms allow-scripts",
            srcDoc: IFRAME_TEMPLATE,
            loading: "lazy",
          })
        )
      ),
    isGenerating &&
    h(
      "div",
      { className: "fixed top-4 left-1/2 -translate-x-1/2 z-10" },
      h(
        "div",
        {
          className:
            "bg-gray-900/80 backdrop-blur-sm rounded-full px-4 py-2 text-sm text-gray-300 flex items-center gap-2",
        },
        h("div", {
          className: "w-2 h-2 bg-green-500 rounded-full animate-pulse",
        }),
        "Generating..."
      )
    ),
    h(
      "div",
      {
        className: `fixed bottom-0 left-0 w-full bg-gray-900/95 backdrop-blur-lg border-t border-gray-800 transition-all duration-300 z-10 ${isCodePanelOpen ? "h-1/2" : "h-0"
          }`,
      },
      isCodePanelOpen &&
      h(React.Fragment, null, [
        h(
          "div",
          {
            className:
              "flex justify-between items-center p-2 border-b border-gray-800",
          },
          h(
            "div",
            { className: "flex items-center gap-2 text-sm text-gray-400" },
            h(Code, { className: "w-4 h-4" }),
            "Generated Code"
          ),
          h(
            "button",
            {
              onClick: () => setIsCodePanelOpen(false),
              className: "p-1 hover:bg-gray-800 rounded",
            },
            h(MinimizeIcon, { className: "w-4 h-4" })
          )
        ),
        h(
          "div",
          { className: "overflow-auto p-4 h-[calc(100%-40px)]" },
          messages.map(
            (message, index) =>
              message.role === "assistant" &&
              message.content &&
              h(
                "pre",
                {
                  key: index,
                  className:
                    "text-sm font-mono text-gray-300 whitespace-pre-wrap",
                },
                message.content
              )
          ),
          error &&
          h(
            "div",
            {
              className: "bg-red-900/50 text-red-300 p-3 rounded-lg mt-4",
            },
            h("strong", null, "Error:"),
            " ",
            error.toString()
          )
        ),
      ])
    ),
    h(
      "div",
      {
        className:
          "fixed bottom-8 left-1/2 -translate-x-1/2 w-full max-w-2xl px-4 z-10",
      },
      h(
        "form",
        { onSubmit: handleSubmit, className: "relative" },
        h(
          "div",
          {
            className:
              "relative flex items-center bg-gray-900/80 backdrop-blur-lg rounded-xl shadow-lg border border-gray-800",
          },
          h("input", {
            type: "text",
            value: input,
            onChange: (e) => setInput(e.target.value),
            placeholder: "Describe what you want to create...",
            className:
              "flex-1 bg-transparent px-4 py-3 focus:outline-none placeholder-gray-500",
            disabled: isGenerating,
          }),
          h(
            "div",
            { className: "flex items-center gap-2 pr-3" },
            !isCodePanelOpen &&
            messages.length > 0 &&
            h(
              "button",
              {
                type: "button",
                className:
                  "p-2 text-gray-400 hover:text-gray-200 transition-colors",
                onClick: () => setIsCodePanelOpen(true),
              },
              h(Code, { className: "w-5 h-5" })
            ),
            h(
              "button",
              {
                type: "submit",
                disabled: isGenerating,
                className:
                  "bg-indigo-500 hover:bg-indigo-600 text-white rounded-lg p-2 transition-colors",
              },
              h(Send, { className: "w-5 h-5" })
            )
          )
        )
      )
    ),
    h("style", { jsx: true, global: true },
      `
        @keyframes fadeIn {
          from {
            opacity: 0;
            transform: scale(0.98);
          }
          to {
            opacity: 1;
            transform: scale(1);
          }
        }
        .animate-fade-in {
          animation: fadeIn 0.5s ease-out forwards;
        }
      `
    )
  );
}
