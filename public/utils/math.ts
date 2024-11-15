export function int64ToFloat16(int64Value) {
  // Convert BigInt to Number (float64)
  const float64Value = Number(int64Value);

  // Handle special cases
  if (!isFinite(float64Value)) return float64Value > 0 ? 0x7c00 : 0xfc00; // +/- infinity
  if (float64Value === 0) return 0; // Zero is represented as 0

  // Get sign, exponent, and mantissa from float64
  const sign = float64Value < 0 ? 1 : 0;
  const absValue = Math.abs(float64Value);
  const exponent = Math.floor(Math.log2(absValue));
  const mantissa = absValue / Math.pow(2, exponent) - 1;

  // Convert exponent and mantissa to float16 format
  const float16Exponent = exponent + 15; // Offset exponent by 15 (float16 bias)
  const float16Mantissa = Math.round(mantissa * 1024); // 10-bit mantissa for float16

  // Handle overflow/underflow
  if (float16Exponent <= 0) {
    // Subnormal numbers (exponent <= 0)
    return (sign << 15) | (float16Mantissa >> 1);
  } else if (float16Exponent >= 31) {
    // Overflow, set to infinity
    return (sign << 15) | 0x7c00;
  } else {
    // Normalized numbers
    return (sign << 15) | (float16Exponent << 10) | (float16Mantissa & 0x3ff);
  }
}

export function float16ToInt64(float16Value: number): bigint {
  // Extract components from float16
  const sign = (float16Value & 0x8000) >> 15;
  const exponent = (float16Value & 0x7c00) >> 10;
  const mantissa = float16Value & 0x03ff;

  // Handle special cases
  if (exponent === 0 && mantissa === 0) return BigInt(0); // Zero
  if (exponent === 0x1f) return sign ? BigInt("-Infinity") : BigInt("Infinity"); // Infinity

  // Convert back to number
  let value;
  if (exponent === 0) {
    // Subnormal numbers
    value = Math.pow(2, -14) * (mantissa / 1024);
  } else {
    // Normalized numbers
    value = Math.pow(2, exponent - 15) * (1 + mantissa / 1024);
  }

  // Apply sign
  value = sign ? -value : value;

  return BigInt(Math.round(value));
}

export function uint16ToFloat16(uint16) {
  // Ensure the input is within the uint16 range
  if (uint16 < 0 || uint16 > 0xffff) {
    throw new RangeError("Input must be a uint16 value (0 to 65535).");
  }

  const sign = (uint16 >> 15) & 0x1; // Extract sign bit
  const exponent = (uint16 >> 10) & 0x1f; // Extract exponent bits
  const mantissa = uint16 & 0x3ff; // Extract mantissa bits

  // Handle special cases for float16
  if (exponent === 0 && mantissa === 0) {
    return 0; // Zero
  } else if (exponent === 31) {
    return mantissa === 0 ? Infinity : NaN; // Infinity or NaN
  }

  // Adjust exponent for float32 representation
  const adjustedExponent = exponent === 0 ? 0 : exponent + (127 - 15);
  const float32 = (sign << 31) | (adjustedExponent << 23) | (mantissa << 13);

  // Convert to a Float32 number
  const buffer = new ArrayBuffer(4);
  new Uint32Array(buffer)[0] = float32;
  return new Float32Array(buffer)[0];
}

function toFloat16(num) {
  const buffer = new ArrayBuffer(4);
  const dataView = new DataView(buffer);
  dataView.setFloat32(0, num, true);
  const float16Bits = dataView.getUint16(0, true);
  return [num, float16Bits];
}

function floatToFloat16(value: number): number {
  if (value === 0) return 0;
  if (isNaN(value)) return 0x7e00;

  const sign = value < 0 ? 0x8000 : 0;
  value = Math.abs(value);

  if (!isFinite(value)) return sign | 0x7c00;

  let exponent = Math.floor(Math.log2(value));
  let fraction = Math.round((value / Math.pow(2, exponent) - 1) * 1024);

  if (exponent <= -15) {
    fraction = Math.round((value / Math.pow(2, -14)) * 1024);
    exponent = 0;
  } else {
    exponent += 15;
    if (exponent >= 31) return sign | 0x7c00;
  }

  return sign | (exponent << 10) | fraction;
}

function float16ToUint16(value: number): number {
  if (value === 0) return 0;
  if (isNaN(value)) return 0x7e00;
  if (!isFinite(value)) return value < 0 ? 0xfc00 : 0x7c00;

  const sign = value < 0 ? 0x8000 : 0;
  value = Math.abs(value);

  let exponent = Math.floor(Math.log2(value));
  let fraction = Math.round((value / Math.pow(2, exponent) - 1) * 1024);

  if (exponent <= -15) {
    fraction = Math.round((value / Math.pow(2, -14)) * 1024);
    exponent = 0;
  } else {
    exponent += 15;
    if (exponent >= 31) return sign | 0x7c00;
  }

  return sign | (exponent << 10) | fraction;
}
