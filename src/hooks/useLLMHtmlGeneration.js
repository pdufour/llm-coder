import React from "react";
import {
  useLLMGeneration,
} from "./useLLMGeneration.js";

export const SYSTEM_PROMPT = `You are a specialized HTML generator that creates complete, production-ready HTML using ONLY Tailwind CSS classes.

IMPORTANT:
- Use EXCLUSIVELY Tailwind CSS classes for ALL styling - NO style attributes or style tags
- Every visual styling must be done through Tailwind classes
- Never use inline styles, CSS, or style tags
- If you need a style that seems custom, use Tailwind's arbitrary value syntax like [w-123px]
- Always write out the full, actual HTML elements with real content
- Never use Lorem Ipsum - write realistic English content
- Never include code comments or explanations
- DO NOT wrap the output in markdown code blocks

STRICT STYLING RULES:
- ❌ NO <style> tags
- ❌ NO style="" attributes
- ❌ NO CSS classes that aren't Tailwind
- ✅ ONLY use Tailwind utility classes
- ✅ Use Tailwind's arbitrary value syntax for custom values
- ✅ Stack multiple Tailwind classes to achieve desired styling

Requirements:
- Generate ONLY the HTML content for inside the <body> tag
- EXCLUSIVELY use Tailwind CSS classes for ALL styling
- NO placeholders, NO comments, NO explanations
- NO server-side code or JavaScript
- Every element must be complete with real content
- Write out all actual HTML elements - never indicate "pending" or "todo" sections
- DO NOT wrap the output in markdown code blocks

Example of CORRECT styling:
<div class="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded">
Instead of:
<div style="background: blue; color: white;">

Generate complete, functional HTML using ONLY Tailwind classes for styling.`;

function cleanHtmlCode(fullHtml) {
  // First remove markdown code blocks
  let cleaned = fullHtml
    .replace(/```html/gi, "")
    .replace(/```(\s+)?html/gi, "")
    .replace(/```(\n)?html/gi, "")
    .replace(/```/g, "")
    .trim();

  // If the content is already wrapped in our template, extract just the content
  const templateContentMatch = cleaned.match(
    /CONTENT_PLACEHOLDER">\s*([\s\S]*?)\s*<\/body>/i
  );
  if (templateContentMatch) {
    return templateContentMatch[1].trim();
  }

  // Try to extract content between body tags
  const bodyContent = cleaned.match(/<body[^>]*>\s*([\s\S]*?)\s*<\/body>/i);
  if (bodyContent) {
    // Further clean the body content of any nested body tags
    return bodyContent[1]
      .replace(/<\/?body[^>]*>/gi, "") // Remove any nested body tags
      .trim();
  }

  // If no body tags found, remove any DOCTYPE and html wrapper if present
  cleaned = cleaned
    .replace(/<!DOCTYPE[^>]*>/gi, "")
    .replace(/<\/?html[^>]*>/gi, "")
    .replace(/<\/?body[^>]*>/gi, "")
    .replace(/<head>[\s\S]*?<\/head>/gi, "")
    .trim();

  return cleaned;
}

export function useLLMHtmlGeneration({
  modelConfig,
  backend,
  systemPrompt = SYSTEM_PROMPT,
}) {
  const [generatedCode, setGeneratedCode] = React.useState("");
  const lastGeneratedCode = React.useRef("");

  const { generate, warmup, isGenerating, error, partialText } = useLLMGeneration(
    modelConfig,
    systemPrompt,
    backend
  );

  React.useEffect(() => {
    if (partialText) {
      const cleaned = cleanHtmlCode(partialText);
      const processedCode = cleaned;
      setGeneratedCode(processedCode);
      lastGeneratedCode.current = cleaned; // Store just the body content
    }
  }, [partialText]);

  const generateCode = React.useCallback(
    async (prompt, extras, config) => {
      const fullPrompt = lastGeneratedCode.current
        ? `Current HTML: \n${lastGeneratedCode.current}\n\nRequest: ${prompt}`
        : `Generate the HTML for: ${prompt}`;

      await generate(fullPrompt, extras, config);
    },
    [generate]
  );

  return {
    generateCode,
    isGenerating,
    error,
    generatedCode,
    warmup,
  };
}
