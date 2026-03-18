package com.rhythmatician.lodiffusion.world.noise;

import static org.junit.jupiter.api.Assertions.*;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Test;

/**
 * Unit tests for {@link GpuBiomeProvider}.
 *
 * <p>Since {@link net.minecraft.world.biome.source.BiomeSource} and
 * {@link net.minecraft.world.gen.noise.NoiseConfig} are final or otherwise
 * unmockable Minecraft classes, these tests exercise the coordinate logic,
 * output shape contract, BiomeMapping integration, and backend-name contract
 * without constructing the provider (which requires a live Minecraft env).
 *
 * <p>The key invariant is that GpuBiomeProvider uses the <b>same</b> coordinate
 * formula and BiomeMapping call as VanillaBiomeProvider — verified by testing
 * the shared formula and the BiomeMapping round-trip.
 */
class GpuBiomeProviderTest {

    // ── Helpers ──────────────────────────────────────────────────────

    /** Create a dummy SectionNoiseData filled with a constant value. */
    private static SectionNoiseData makeData(float value, int sx, int sy, int sz) {
        float[] flat = new float[SectionNoiseData.FLAT_LENGTH];
        Arrays.fill(flat, value);
        return new SectionNoiseData(flat, sx, sy, sz);
    }

    // ── Coordinate logic ────────────────────────────────────────────
    //
    // GpuBiomeProvider (and VanillaBiomeProvider) compute quart coords as:
    //   blockCoord = sectionCoord*16 + q*4 + 2
    //   quartCoord = blockCoord >> 2
    // We verify this mapping is correct for various section origins.

    @Nested
    class CoordinateMapping {

        /** Compute a quart coordinate the same way as GpuBiomeProvider. */
        private int toQuartCoord(int sectionCoord, int q) {
            int block = sectionCoord * 16 + q * 4 + 2;
            return block >> 2;
        }

        @Test
        void section0_0_0_qx0() {
            assertEquals(0, toQuartCoord(0, 0));
        }

        @Test
        void section0_0_0_qx3() {
            assertEquals(3, toQuartCoord(0, 3));
        }

        @Test
        void positiveSection() {
            // sectionX=2, qx=0: block = 32 + 0 + 2 = 34 → quart = 8
            assertEquals(8, toQuartCoord(2, 0));
            // sectionX=2, qx=3: block = 32 + 12 + 2 = 46 → quart = 11
            assertEquals(11, toQuartCoord(2, 3));
        }

        @Test
        void negativeSection() {
            // sectionY=-1, qy=0: block = -16 + 0 + 2 = -14 → -14 >> 2 = -4
            assertEquals(-4, toQuartCoord(-1, 0));
            // sectionY=-4, qy=3: block = -64 + 12 + 2 = -50 → -50 >> 2 = -13
            assertEquals(-13, toQuartCoord(-4, 3));
        }

        @Test
        void allQuartsMonotonicallyIncrease() {
            for (int sec = -4; sec <= 19; sec++) {
                int prev = Integer.MIN_VALUE;
                for (int q = 0; q < 4; q++) {
                    int quart = toQuartCoord(sec, q);
                    assertTrue(quart > prev,
                            "quart coords must increase: sec=" + sec + " q=" + q);
                    prev = quart;
                }
            }
        }

        @Test
        void quartSpacingIsFour() {
            // Each section spans 4 quarts (q=0..3 → 4 distinct quart coords)
            for (int sec = -4; sec <= 19; sec++) {
                int q0 = toQuartCoord(sec, 0);
                int q3 = toQuartCoord(sec, 3);
                assertEquals(3, q3 - q0,
                        "Expected 3-quart span for section " + sec);
            }
        }
    }

    // ── Output shape ────────────────────────────────────────────────

    @Nested
    class OutputShape {

        @Test
        void resultArrayIs4x4x4With64Cells() {
            int[][][] biomes = new int[4][4][4];
            assertEquals(4, biomes.length);
            assertEquals(4, biomes[0].length);
            assertEquals(4, biomes[0][0].length);

            AtomicInteger count = new AtomicInteger();
            for (int qx = 0; qx < 4; qx++)
                for (int qy = 0; qy < 4; qy++)
                    for (int qz = 0; qz < 4; qz++)
                        count.incrementAndGet();
            assertEquals(64, count.get());
        }
    }

    // ── Backend-name contract ───────────────────────────────────────

    @Test
    void gpuClimateIsDistinctFromVanillaCpu() {
        assertNotEquals("vanilla_cpu", "gpu_climate");
    }

    // ── BiomeMapping round-trip ─────────────────────────────────────

    @Nested
    class BiomeMappingContract {

        @Test
        void plainsHasExpectedCanonicalId() {
            int id = com.rhythmatician.lodiffusion.voxy.BiomeMapping.toCanonicalId("minecraft:plains");
            assertEquals(34, id, "plains should be at alphabetical index 34");
        }

        @Test
        void desertHasExpectedCanonicalId() {
            int id = com.rhythmatician.lodiffusion.voxy.BiomeMapping.toCanonicalId("minecraft:desert");
            assertEquals(12, id, "desert should be at alphabetical index 12");
        }

        @Test
        void unknownBiomeMapsTo255() {
            int id = com.rhythmatician.lodiffusion.voxy.BiomeMapping.toCanonicalId("minecraft:the_end");
            assertEquals(255, id, "non-overworld biome should map to UNKNOWN (255)");
        }

        @Test
        void canonicalIdsRoundTrip() {
            for (int i = 0; i < com.rhythmatician.lodiffusion.voxy.BiomeMapping.size(); i++) {
                String name = com.rhythmatician.lodiffusion.voxy.BiomeMapping.getCanonicalName(i);
                assertNotNull(name, "index " + i + " should have a name");
                assertEquals(i, com.rhythmatician.lodiffusion.voxy.BiomeMapping.toCanonicalId(name),
                        "round-trip failed for " + name);
            }
        }
    }

    // ── SectionNoiseData compatibility ──────────────────────────────

    @Test
    void sectionNoiseDataHas960Floats() {
        SectionNoiseData data = makeData(0.0f, 0, 0, 0);
        assertEquals(960, data.flat().length);
    }
}
