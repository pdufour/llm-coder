import React from "react";
import { useLLMGeneration } from "./useLLMGeneration.js";

export const SYSTEM_PROMPT = `You are a specialized HTML generator that creates complete, production-ready HTML using ONLY Tailwind CSS classes.`;

export function useLLMVisionGeneration({
  modelConfig,
  backend = 'qwen2vl',
  systemPrompt = SYSTEM_PROMPT,
}) {
  const [generatedText, setGeneratedText] = React.useState("");
  const lastGeneratedText = React.useRef("");

  const { generate, isGenerating, error, partialText } = useLLMGeneration(
    modelConfig,
    systemPrompt,
    backend,
  );

  React.useEffect(() => {
    if (partialText) {
      console.log({ partialText });
      setGeneratedText(partialText);
      lastGeneratedText.current = partialText;
    }
  }, [partialText]);

  const generateText = React.useCallback(
    async (prompt, extras) => {
      const fullPrompt = lastGeneratedText.current
        ? `Current Chat: \n${lastGeneratedCode.current}\n\nRequest: ${prompt}`
        : `${prompt} `;

      await generate(fullPrompt, extras);
    },
    [generate]
  );

  return {
    generateText,
    isGenerating,
    error,
    generatedText,
  };
}
