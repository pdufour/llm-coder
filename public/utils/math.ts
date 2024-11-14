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

export function uint16ToFloat16(value: number) {
  const sign = (value & 0x8000) >> 15;
  const exponent = (value & 0x7c00) >> 10;
  const fraction = value & 0x03ff;

  if (exponent === 0) {
    if (fraction === 0) return 0;
    return (sign ? -1 : 1) * Math.pow(2, -14) * (fraction / 1024);
  } else if (exponent === 0x1f) {
    // Special handling for fraction in this case
    return (sign ? -1 : 1) * Math.pow(2, 16) * (1 + fraction / 1024);
  }

  return (sign ? -1 : 1) * Math.pow(2, exponent - 15) * (1 + fraction / 1024);
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
