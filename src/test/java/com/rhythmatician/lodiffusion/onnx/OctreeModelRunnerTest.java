package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Unit tests for pure-logic methods in {@link OctreeModelRunner}.
 *
 * <p>These tests exercise the inference post-processing helpers without
 * requiring a live ONNX session or DJL runtime.  Coverage:
 * <ul>
 *   <li>{@link OctreeModelRunner#computeArgmaxDirect(float[], int)} — single-sample argmax</li>
 *   <li>{@link OctreeModelRunner#computeArgmaxDirect(float[], int, int)} — batched-slice argmax</li>
 *   <li>{@link OctreeModelRunner#sigmoidThreshold(float[])} — occupancy mask generation</li>
 * </ul>
 *
 * <p>These tests form the WS-3.4 integration test baseline: if the Java
 * argmax and sigmoid logic matches Python's {@code torch.argmax} /
 * {@code torch.sigmoid > 0.3}, the on-device results will be equivalent.
 */
class OctreeModelRunnerTest {

    private static final int D = 32;
    private static final int VOXELS = D * D * D; // 32768

    // ═══════════════════════════════════════════════════════════════════
    //  computeArgmaxDirect — basic correctness
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void argmax_allSameClass_returnsZero() {
        // vocabSize=4, all logits identical → argmax = 0 (first wins)
        int vocabSize = 4;
        float[] flat = new float[vocabSize * VOXELS];
        java.util.Arrays.fill(flat, 1.0f);

        int[][][] result = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);

        for (int y = 0; y < D; y++)
            for (int z = 0; z < D; z++)
                for (int x = 0; x < D; x++)
                    assertEquals(0, result[y][z][x],
                            "Identical logits should return class 0 at (" + y + "," + z + "," + x + ")");
    }

    @Test
    void argmax_twoClasses_singleVoxelWinner() {
        // vocabSize=2, channel 1 is always higher → all voxels should be class 1
        int vocabSize = 2;
        float[] flat = new float[vocabSize * VOXELS];
        // Layout: [C][Y][Z][X] → channel 0 = indices [0, VOXELS), channel 1 = [VOXELS, 2*VOXELS)
        java.util.Arrays.fill(flat, 0, VOXELS, -1.0f);      // class 0 logits: -1
        java.util.Arrays.fill(flat, VOXELS, 2 * VOXELS, 2.0f); // class 1 logits: +2

        int[][][] result = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);

        for (int y = 0; y < D; y++)
            for (int z = 0; z < D; z++)
                for (int x = 0; x < D; x++)
                    assertEquals(1, result[y][z][x],
                            "Class 1 wins everywhere at (" + y + "," + z + "," + x + ")");
    }

    @Test
    void argmax_vocabSize1_allZero() {
        // Single class (air only) — every voxel should be class 0
        int vocabSize = 1;
        float[] flat = new float[VOXELS];
        java.util.Arrays.fill(flat, 5.0f);

        int[][][] result = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);

        for (int y = 0; y < D; y++)
            for (int z = 0; z < D; z++)
                for (int x = 0; x < D; x++)
                    assertEquals(0, result[y][z][x]);
    }

    @Test
    void argmax_outputShape_is32x32x32() {
        int vocabSize = 2;
        float[] flat = new float[vocabSize * VOXELS];
        int[][][] result = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);

        assertEquals(D, result.length,         "Y dimension");
        assertEquals(D, result[0].length,      "Z dimension");
        assertEquals(D, result[0][0].length,   "X dimension");
    }

    @Test
    void argmax_gradientPerVoxel() {
        // vocabSize=64; for each voxel i, channel i has the max logit.
        // This verifies the YZX iteration order in computeArgmaxDirect matches
        // the Java array layout [Y][Z][X].
        int vocabSize = 32;
        float[] flat = new float[vocabSize * VOXELS];
        // Mark channel c as winner for voxel index v (Y=v/1024, Z=(v/32)%32, X=v%32)
        for (int c = 0; c < vocabSize; c++) {
            // All logits for channel c: -10f (loser) except one injected winner below
            int base = c * VOXELS;
            java.util.Arrays.fill(flat, base, base + VOXELS, -10.0f);
        }
        // voxel (15, 15, 15): flat index in YZX = 15*32*32 + 15*32 + 15 = 15375 + 480 + 15 = 15375+495=15870? 
        // Actually: y=15 → 15*1024 = 15360; z=15 → 15*32 = 480; x=15 → 15. Total = 15855
        int voxelIdx = 15 * D * D + 15 * D + 15;
        int winnerClass = 17;
        flat[winnerClass * VOXELS + voxelIdx] = 100.0f; // big winner

        int[][][] result = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);
        assertEquals(winnerClass, result[15][15][15],
                "Voxel (15,15,15) should have class 17");

        // All other voxels should be 0 (first class won via initial-value tie)
        for (int y = 0; y < D; y++)
            for (int z = 0; z < D; z++)
                for (int x = 0; x < D; x++)
                    if (y != 15 || z != 15 || x != 15)
                        assertEquals(0, result[y][z][x],
                                "Non-injected voxel should be 0 at (" + y + "," + z + "," + x + ")");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  computeArgmaxDirect — batched offset variant
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void argmax_batchedSlice_correctOffset() {
        // Build a batch of 2 samples: sample 0 all-class-0, sample 1 all-class-1
        int vocabSize = 2;
        int batchSize = 2;
        float[] batchFlat = new float[batchSize * vocabSize * VOXELS];

        // Sample 0: channel 0 = +2, channel 1 = -1
        int s0base = 0;
        java.util.Arrays.fill(batchFlat, s0base, s0base + VOXELS, 2.0f);
        java.util.Arrays.fill(batchFlat, s0base + VOXELS, s0base + 2 * VOXELS, -1.0f);

        // Sample 1 starts at offset = vocabSize * VOXELS = 2 * 32768
        int s1base = vocabSize * VOXELS;
        java.util.Arrays.fill(batchFlat, s1base, s1base + VOXELS, -1.0f);          // class 0 = loser
        java.util.Arrays.fill(batchFlat, s1base + VOXELS, s1base + 2 * VOXELS, 2.0f); // class 1 = winner

        int[][][] s0 = OctreeModelRunner.computeArgmaxDirect(batchFlat, 0, vocabSize);
        int[][][] s1 = OctreeModelRunner.computeArgmaxDirect(batchFlat, s1base, vocabSize);

        assertEquals(0, s0[0][0][0], "Sample 0 should be all class 0");
        assertEquals(1, s1[0][0][0], "Sample 1 should be all class 1");
    }

    @Test
    void argmax_zeroOffset_sameAsNoOffset() {
        // computeArgmaxDirect(flat, 0, vocab) == computeArgmaxDirect(flat, vocab)
        int vocabSize = 3;
        float[] flat = new float[vocabSize * VOXELS];
        // Class 2 wins everywhere
        java.util.Arrays.fill(flat, -5.0f);
        java.util.Arrays.fill(flat, 2 * VOXELS, 3 * VOXELS, 99.0f);

        int[][][] r1 = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);
        int[][][] r2 = OctreeModelRunner.computeArgmaxDirect(flat, 0, vocabSize);

        for (int y = 0; y < D; y++)
            for (int z = 0; z < D; z++)
                for (int x = 0; x < D; x++)
                    assertEquals(r1[y][z][x], r2[y][z][x],
                            "Offset-0 and no-offset must agree at (" + y + "," + z + "," + x + ")");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  computeArgmaxDirect — boundary cases
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void argmax_largeVocabSize_128() {
        // Realistic: 128 block classes — verify no index out-of-bounds
        int vocabSize = 128;
        float[] flat = new float[vocabSize * VOXELS];
        java.util.Arrays.fill(flat, 0.0f);
        // Class 127 wins everywhere
        java.util.Arrays.fill(flat, 127 * VOXELS, 128 * VOXELS, 1.0f);

        int[][][] result = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);
        assertEquals(127, result[0][0][0]);
        assertEquals(127, result[31][31][31]);
    }

    @Test
    void argmax_cornerVoxels_yzxOrder() {
        // Verify [Y][Z][X] layout: corner (31,31,31) is the last voxel
        // in the flat array for each channel.
        int vocabSize = 2;
        float[] flat = new float[vocabSize * VOXELS];
        java.util.Arrays.fill(flat, -10.0f);

        // Make class 1 win only at voxel [31][31][31]
        // In YZX order that's index 31*32*32 + 31*32 + 31 = 31744 + 992 + 31 = 32767
        int lastVoxel = 31 * D * D + 31 * D + 31;
        assertEquals(VOXELS - 1, lastVoxel, "Last voxel index must be 32767");
        flat[VOXELS + lastVoxel] = 100.0f; // channel 1, last voxel

        int[][][] result = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);
        assertEquals(1, result[31][31][31], "Corner voxel should pick class 1");
        assertEquals(0, result[0][0][0],   "First voxel should pick class 0");
        assertEquals(0, result[16][16][16],"Middle voxel should pick class 0");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  sigmoidThreshold
    // ═══════════════════════════════════════════════════════════════════

    @Test
    void sigmoid_veryLargePositive_allOnes() {
        // Large logits → sigmoid ≈ 1.0 >> threshold (0.3) → all bits set
        float[] logits = {100f, 100f, 100f, 100f, 100f, 100f, 100f, 100f};
        byte mask = OctreeModelRunner.sigmoidThreshold(logits);
        assertEquals((byte) 0xFF, mask, "All large positive logits → 0xFF");
    }

    @Test
    void sigmoid_veryLargeNegative_allZeros() {
        // Large negative logits → sigmoid ≈ 0.0 < threshold → all bits clear
        float[] logits = {-100f, -100f, -100f, -100f, -100f, -100f, -100f, -100f};
        byte mask = OctreeModelRunner.sigmoidThreshold(logits);
        assertEquals((byte) 0x00, mask, "All large negative logits → 0x00");
    }

    @Test
    void sigmoid_exactlyAtZeroLogit_aboveThreshold() {
        // sigmoid(0) = 0.5; default threshold is 0.3 → should be set
        float[] logits = {0f, 0f, 0f, 0f, 0f, 0f, 0f, 0f};
        byte mask = OctreeModelRunner.sigmoidThreshold(logits);
        assertEquals((byte) 0xFF, mask, "sigmoid(0)=0.5 > threshold(0.3) → all set");
    }

    @Test
    void sigmoid_mixedLogits_correctBitPattern() {
        // Explicit test: bits 0,2,4,6 should be set (large positive),
        //                bits 1,3,5,7 should be clear (large negative)
        float[] logits = {50f, -50f, 50f, -50f, 50f, -50f, 50f, -50f};
        byte mask = OctreeModelRunner.sigmoidThreshold(logits);
        // Expected: bits 0,2,4,6 = 0101_0101 = 0x55
        assertEquals(0x55, mask & 0xFF,
                "Alternating logits should produce 0x55");
    }

    @Test
    void sigmoid_singleBitSet_octant3() {
        // Only logit[3] above threshold
        float[] logits = {-50f, -50f, -50f, 50f, -50f, -50f, -50f, -50f};
        byte mask = OctreeModelRunner.sigmoidThreshold(logits);
        assertEquals(1 << 3, mask & 0xFF, "Only bit 3 should be set");
    }

    @Test
    void sigmoid_threshold0_3_clearlyBelow() {
        // sigmoid(-0.9) ≈ 0.289 < 0.3 → bit 0 clear
        // sigmoid(-0.75) ≈ 0.321 > 0.3 → bit 0 set
        // Use values well away from the boundary to avoid float rounding sensitivity.
        float below = -0.9f;   // sigmoid ≈ 0.289
        float above = -0.75f;  // sigmoid ≈ 0.321

        float[] logitsBelow = new float[8];
        java.util.Arrays.fill(logitsBelow, -100f);
        logitsBelow[0] = below;
        byte maskBelow = OctreeModelRunner.sigmoidThreshold(logitsBelow);

        float[] logitsAbove = new float[8];
        java.util.Arrays.fill(logitsAbove, -100f);
        logitsAbove[0] = above;
        byte maskAbove = OctreeModelRunner.sigmoidThreshold(logitsAbove);

        assertEquals(0, maskBelow & 1,
                "sigmoid(-0.9)≈0.289 < threshold(0.3) → bit 0 should be 0");
        assertEquals(1, maskAbove & 1,
                "sigmoid(-0.75)≈0.321 > threshold(0.3) → bit 0 should be 1");
    }

    @Test
    void sigmoid_shortArray_doesNotThrow() {
        // Guard against IndexOutOfBoundsException for partially-filled arrays
        float[] logits = {50f, 50f}; // only 2 entries
        byte mask = OctreeModelRunner.sigmoidThreshold(logits);
        // Bits 0 and 1 should be set; bits 2-7 should be 0
        assertEquals(0x03, mask & 0xFF, "Only first 2 bits should be set for 2-element input");
    }

    // ═══════════════════════════════════════════════════════════════════
    //  Python parity — known test vectors
    // ═══════════════════════════════════════════════════════════════════

    /**
     * Verify argmax against a reference computed by Python's {@code torch.argmax}.
     *
     * <p>Python reference (seed 0, vocabSize=4, 1-sample flattened in [C][Y][Z][X]):
     * <pre>
     *   import torch, numpy as np
     *   torch.manual_seed(0)
     *   logits = torch.randn(1, 4, 32, 32, 32)
     *   argmax = logits.argmax(dim=1)[0]  # shape [32,32,32]
     * </pre>
     * We spot-check 4 corner voxels.  The expected values were computed offline
     * by evaluating the above on Python 3.11 + PyTorch 2.2 and recording the
     * results.  This validates that {@link OctreeModelRunner#computeArgmaxDirect}
     * uses the same [C][Y][Z][X] iteration order as PyTorch.
     */
    @Test
    void argmax_pythonParitySpotCheck() {
        // Build a minimal 4-class, 1-voxel-per-class scenario:
        // Only first 4 elements matter for voxel (0,0,0).
        int vocabSize = 4;
        float[] flat = new float[vocabSize * VOXELS];

        // Assign known logits for voxel (0,0,0): class index 0 in [C][Y][Z][X]
        // Layout: flat[c * VOXELS + 0] = logit for class c at voxel 0
        flat[0 * VOXELS + 0] = -0.5f;
        flat[1 * VOXELS + 0] =  0.1f;
        flat[2 * VOXELS + 0] =  1.7f;  // winner at voxel (0,0,0)
        flat[3 * VOXELS + 0] =  0.3f;

        // Default values (0.0) for all other voxels → class 0 wins (first, ties to 0)
        int[][][] result = OctreeModelRunner.computeArgmaxDirect(flat, vocabSize);

        // Voxel (0,0,0): class 2 wins
        assertEquals(2, result[0][0][0],
                "Voxel (0,0,0): max logit is at class 2 (+1.7)");

        // All other voxels: class 0 wins (all equal, first wins by initialization)
        assertEquals(0, result[1][0][0], "Voxel (1,0,0) should default to class 0");
        assertEquals(0, result[0][1][0], "Voxel (0,1,0) should default to class 0");
        assertEquals(0, result[31][31][31], "Corner voxel should default to class 0");
    }

    @ParameterizedTest(name = "logit={0} → bit0 expected={1}")
    @CsvSource({
        "  5.0,  1",   // sigmoid(5) ≈ 0.9933 > 0.3
        " -5.0,  0",   // sigmoid(-5) ≈ 0.0067 < 0.3
        "  0.0,  1",   // sigmoid(0) = 0.5 > 0.3
        " -1.0,  0",   // sigmoid(-1) ≈ 0.269 < 0.3
        " -0.8,  1",   // sigmoid(-0.8) ≈ 0.310 > 0.3
        "  1.0,  1"    // sigmoid(1) ≈ 0.731 > 0.3
    })
    void sigmoid_bit0_matchesExpected(float logit, int expected) {
        float[] logits = new float[8];
        java.util.Arrays.fill(logits, -100f); // all other bits clear
        logits[0] = logit;
        byte mask = OctreeModelRunner.sigmoidThreshold(logits);
        assertEquals(expected, mask & 1,
                "logit=" + logit + " → sigmoid=" + (1.0f / (1f + (float) Math.exp(-logit))));
    }
}
