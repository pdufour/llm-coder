class Logger {
  static indentLevel: number = 0;

  static group(...args) {
    logger.indentLevel += 1;
    console.group(...args);
  }

  static groupCollapsed(...args) {
    logger.indentLevel += 1;
    console.groupCollapsed(...args);
  }

  static groupEnd() {
    if (logger.indentLevel === 0) {
      console.error("logGroupEnd() called too many times");
      return;
    }
    logger.indentLevel -= 1;
    console.groupEnd();
  }

  static tensor(name: string, tensor: any) {
    logger.groupCollapsed(`[NUMPY] ${name}:`);
    logger.debug(`Shape: (${tensor.dims.join(", ")})`);
    logger.debug(`Dtype: ${tensor.type}`);

    if (tensor.data) {
      const data = Array.from(tensor.data as (number | bigint)[]).map(
        (val: number | bigint) => {
          return val.toString();
        }
      );

      if (tensor.dims[0] === 1) {
        logger.debug(`Values: [${data[0]}]`);
      }

      logger.debug(`First few values: [${data.slice(0, 5).join(", ")}]`);
      logger.debug(`Last few values: [${data.slice(-5).join(", ")}]`);
    }
    logger.groupEnd();
  }
}

export const logger = new Proxy(Logger, {
  get(target, prop) {
    // If the method exists on Logger, use it
    if (prop in target) {
      return target[prop];
    }

    // If the method doesn’t exist on Logger, proxy it to console
    if (typeof console[prop] === "function") {
      return console[prop].bind(console);
    }

    // If neither Logger nor console has the method, return undefined
    return undefined;
  },
}) as unknown as typeof console & typeof Logger;
