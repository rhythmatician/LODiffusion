package com.rhythmatician.lodiffusion.onnx;

/**
 * Result of a single model inference pass.
 *
 * @param blockLogits raw logits  [1, N, 16, 16, 16] — axis order (batch, vocab, y, z, x)
 * @param airMask     air/solid   [1, 1, 16, 16, 16] — positive = solid
 * @param elapsedMs   wall-clock inference time in milliseconds
 */
public record InferenceResult(
    float[][][][][] blockLogits,
    float[][][][][] airMask,
    long elapsedMs
) {}
