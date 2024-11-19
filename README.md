# llm-coder
llm-coder is a open source privacy-first AI website-builder powered by your browser. No API keys required. No server required. See the live demo at [https://pdufour.github.io/llm-coder/](https://pdufour.github.io/llm-coder/).

![site](https://github.com/pdufour/llm-coder/raw/main/public/site-50.webp)

**Privacy:** All AI models run locally in your browser. Your data never leaves your device - no servers involved.

It supports the following features:
1. Describe a site you want - the tool will generate the HTML and CSS for you
2. Drag and drop an image - the tool will generate the HTML and CSS for you

## Tech Stack
- [onnxruntime-web](https://onnxruntime.ai/docs/tutorials/web/)

## Setup

1. Clone the repository
2. Install [static-web-server](https://static-web-server.net/download-and-install/#nixos)
2. Run `make run`
3. Visit http://localhost:3000/llm-coder in the browser
