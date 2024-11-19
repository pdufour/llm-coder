import {
  AutoModel,
  AutoTokenizer,
  TextStreamer
} from "@huggingface/transformers"
import { CreateMLCEngine } from "@mlc-ai/web-llm"
import React from "react"

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
      callbacks.onError(error)
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
    async prompt => {
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
        await backendRef.current.generate(prompt, systemPrompt, callbacks)
      } catch (err) {
        callbacks.onError(err)
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
