import { Float16Array, getFloat16, setFloat16 } from "@petamoriken/float16";

class Logger {
  static indentLevel = 0

  static group(...args) {
    logger.indentLevel += 1
    console.group(...args)
  }

  static groupCollapsed(...args) {
    logger.indentLevel += 1
    console.groupCollapsed(...args)
  }

  static groupEnd() {
    if (logger.indentLevel === 0) {
      console.error("logGroupEnd() called too many times")
      return
    }
    logger.indentLevel -= 1
    console.groupEnd()
  }

  static tensor(name, tensor) {
    logger.groupCollapsed(`[NUMPY] ${name}:`)
    logger.debug(`Shape: (${tensor.dims.join(", ")})`)
    logger.debug(`Dtype: ${tensor.type}`)

    let data
    if (tensor.data) {
      if (tensor.type === "float16") {
        const view = new DataView(
          tensor.data.buffer,
          tensor.data.byteOffset,
          tensor.data.byteLength
        )
        const littleEndian = true // Adjust if necessary

        const numElements = tensor.data.length
        data = []
        for (let i = 0; i < numElements; i++) {
          const offset = i * 2
          const value = getFloat16(view, offset, littleEndian)
          data.push(value)
        }
      } else {
        data = Array.from(tensor.data).map(val => {
          return val.toString()
        })
      }

      if (tensor.dims[0] === 1) {
        logger.debug(`Values: [${data[0]}]`)
      }

      logger.debug(`First few values: [${data.slice(0, 5).join(", ")}]`)
      logger.debug(`Last few values: [${data.slice(-5).join(", ")}]`)
    }
    logger.groupEnd()
  }
}

export const logger = new Proxy(Logger, {
  get(target, prop) {
    // If the method exists on Logger, use it
    if (prop in target) {
      return target[prop]
    }

    // If the method doesn’t exist on Logger, proxy it to console
    if (typeof console[prop] === "function") {
      return console[prop].bind(console)
    }

    // If neither Logger nor console has the method, return undefined
    return undefined
  }
})
