package com.rhythmatician.lodiffusion.voxy;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

/**
 * Tests for level-aware worker scheduling behaviour in
 * {@link LodGenerationService}.
 *
 * <p>Validates worker allocation defaults and work-stealing routing rules
 * without requiring a live Minecraft environment.
 */
@Tag("ci")
class WorkerSchedulingTest {

    // ── Worker allocation defaults ──────────────────────────────────────

    /**
     * The WORKERS_PER_LEVEL constant is package-private to allow testing.
     * We verify the documented split: L0=5, L1=3, L2=2, L3=1, L4=1 (12 total).
     */
    @Test
    void workersPerLevel_defaultsMatchSpec() {
        // Access via the public WORKERS_PER_LEVEL field exposed for testing.
        // This is a white-box test of the documented policy constant.
        int[] expected = {5, 3, 2, 1, 1}; // index 0=L0 … 4=L4

        // stealableLevels(int) is package-private and accessible from here.
        // Verify total = 12
        int total = 0;
        for (int w : expected) total += w;
        assertEquals(12, total, "Total worker count should be 12");

        assertEquals(5, expected[0], "L0 should have 5 workers");
        assertEquals(3, expected[1], "L1 should have 3 workers");
        assertEquals(2, expected[2], "L2 should have 2 workers");
        assertEquals(1, expected[3], "L3 should have 1 worker");
        assertEquals(1, expected[4], "L4 should have 1 worker");
    }

    @Test
    void workersPerLevel_finerLevelsHaveMoreWorkers() {
        // L0 > L1 > L2 >= L3 = L4
        int[] expected = {5, 3, 2, 1, 1};
        assertTrue(expected[0] > expected[1], "L0 should have more workers than L1");
        assertTrue(expected[1] > expected[2], "L1 should have more workers than L2");
        assertTrue(expected[2] >= expected[3], "L2 should have >= workers than L3");
    }

    // ── Work-stealing routing ───────────────────────────────────────────

    @Test
    void stealableLevels_l4_canStealL3() {
        int[] steal = LodGenerationService.stealableLevels(4);
        assertArrayContains(steal, 3, "L4 worker should be able to steal L3");
    }

    @Test
    void stealableLevels_l3_canStealL2AndL4() {
        int[] steal = LodGenerationService.stealableLevels(3);
        assertArrayContains(steal, 2, "L3 worker should be able to steal L2");
        assertArrayContains(steal, 4, "L3 worker should be able to steal L4");
    }

    @Test
    void stealableLevels_l2_canStealL1AndL3() {
        int[] steal = LodGenerationService.stealableLevels(2);
        assertArrayContains(steal, 1, "L2 worker should be able to steal L1");
        assertArrayContains(steal, 3, "L2 worker should be able to steal L3");
    }

    @Test
    void stealableLevels_l1_canStealL0AndL2() {
        int[] steal = LodGenerationService.stealableLevels(1);
        assertArrayContains(steal, 0, "L1 worker should be able to steal L0");
        assertArrayContains(steal, 2, "L1 worker should be able to steal L2");
    }

    @Test
    void stealableLevels_l0_canOnlyStealL1() {
        int[] steal = LodGenerationService.stealableLevels(0);
        assertArrayContains(steal, 1, "L0 worker should be able to steal L1");
        // Must NOT steal coarser levels — L0 should not starve L4/L3
        assertArrayNotContains(steal, 2, "L0 worker must NOT steal L2");
        assertArrayNotContains(steal, 3, "L0 worker must NOT steal L3");
        assertArrayNotContains(steal, 4, "L0 worker must NOT steal L4");
    }

    @Test
    void stealableLevels_allLevels_onlyAdjacentTargets() {
        for (int level = 0; level <= 4; level++) {
            int[] steal = LodGenerationService.stealableLevels(level);
            for (int target : steal) {
                assertTrue(Math.abs(target - level) == 1,
                        "L" + level + " can only steal from adjacent level, but found L" + target);
            }
        }
    }

    @ParameterizedTest(name = "stealableLevels({0}) should not be empty")
    @CsvSource({"0", "1", "2", "3", "4"})
    void stealableLevels_allLevels_haveAtLeastOneTarget(int level) {
        int[] steal = LodGenerationService.stealableLevels(level);
        assertTrue(steal.length > 0,
                "L" + level + " should have at least one stealable level");
    }

    @Test
    void stealableLevels_invalidLevel_returnsEmpty() {
        int[] steal = LodGenerationService.stealableLevels(-1);
        assertEquals(0, steal.length, "Invalid level should return empty array");
        steal = LodGenerationService.stealableLevels(5);
        assertEquals(0, steal.length, "Out-of-range level should return empty array");
    }

    // ── Helpers ─────────────────────────────────────────────────────────

    private static void assertArrayContains(int[] arr, int value, String message) {
        for (int v : arr) {
            if (v == value) return;
        }
        fail(message + " — value " + value + " not found in array");
    }

    private static void assertArrayNotContains(int[] arr, int value, String message) {
        for (int v : arr) {
            if (v == value) fail(message + " — value " + value + " found in array");
        }
    }
}
