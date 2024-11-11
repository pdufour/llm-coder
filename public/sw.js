import { registerRoute } from "https://cdn.jsdelivr.net/npm/workbox-routing/+esm";
import { NetworkFirst } from "https://cdn.jsdelivr.net/npm/workbox-strategies/+esm";

import initSwc, {
  transformSync,
} from "https://cdn.jsdelivr.net/npm/@swc/wasm-web/+esm";
import "https://cdn.jsdelivr.net/npm/workbox-sw/+esm";

workbox.setConfig({
  debug: true,
});

self.addEventListener("install", function (event) {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", function (event) {
  event.waitUntil(self.clients.claim());
});

let swcInitialized = false;

registerRoute(
  ({ url }) => url.pathname.match(/\.(tsx?|jsx)$/),
  async ({ request }) => {
    const response = await fetch(request);

    if (!swcInitialized) {
      await initSwc();
      swcInitialized = true;
    }

    const source = await response.text();

    const { code } = transformSync(source, {
      jsc: {
      "externalHelpers": true,
        loose: true,
        parser: { syntax: "typescript", tsx: true },
        target: "es2022",
        transform: {
          react: {
            development: true,
          },
        },
      },
      module: {
        type: "es6",
        interop: "node"  // Or "swc" or "none"
      },
    });

    return new Response(code, {
      headers: {
        "Content-Type": "application/javascript",
        "Cache-Control": "no-cache",
      },
    });
  }
);
