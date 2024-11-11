import { encodeSHA256 } from "./crypto.ts";

export class CSPBuilder {
  policy: { [directive: string]: Set<string> };
  private nonce?: string;

  constructor() {
    this.policy = {};
  }

  addDirective(directive: string, source: string) {
    if (!this.policy[directive]) {
      this.policy[directive] = new Set();
    }
    this.policy[directive].add(source);
    return this; // Enable chaining
  }

  async addScriptHash(script: string) {
    const hash = await encodeSHA256(script);
    this.addDirective("script-src", `'sha256-${hash}'`);
    return this; // Enable chaining
  }

  addScriptNonce() {
    if (!this.nonce) {
      this.nonce = this.generateNonce();
    }
    this.addDirective("script-src", `'nonce-${this.nonce}'`);
    return this; // Enable chaining
  }

  private generateNonce() {
    const array = new Uint8Array(16); // 128-bit nonce
    crypto.getRandomValues(array);
    return btoa(String.fromCharCode(...array));
  }

  async build(): Promise<{ cspString: string; nonce?: string }> {
    const cspString = Object.entries(this.policy)
      .map(
        ([directive, sources]) =>
          `${directive} ${Array.from(sources).join(" ")}`
      )
      .join("; ");
    return { cspString, nonce: this.nonce }; // Return both CSP and nonce for use in templates
  }
}
