export const encodeSHA256 = async (script) =>
  `${btoa(
    String.fromCharCode(
      ...new Uint8Array(
        await crypto.subtle.digest("SHA-256", new TextEncoder().encode(script))
      )
    )
  )}`;
