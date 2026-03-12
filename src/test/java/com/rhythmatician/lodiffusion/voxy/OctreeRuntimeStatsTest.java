package com.rhythmatician.lodiffusion.voxy;

import static org.junit.jupiter.api.Assertions.*;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

import com.rhythmatician.lodiffusion.util.PerformanceMonitor;

/**
 * Unit tests for {@link OctreeRuntimeStats} snapshot collection and
 * per-level metric aggregation in {@link OctreeQueue}.
 *
 * <p>No ONNX models or Minecraft runtime required — all tests operate on
 * pure Java data structures.
 */
@Tag("ci")
class OctreeRuntimeStatsTest {

    @BeforeEach
    void resetMonitor() {
        PerformanceMonitor.reset();
    }

    // ── queue inspection ────────────────────────────────────────────────

    @Test
    void queueSize_emptyQueue_returnsZero() {
        OctreeQueue queue = new OctreeQueue();
        for (int lvl = 0; lvl < 5; lvl++) {
            assertEquals(0, queue.queueSize(lvl), "L" + lvl + " should be empty");
        }
    }

    @Test
    void queueSize_afterEnqueueRoot_returnsOne() {
        OctreeQueue queue = new OctreeQueue();
        OctreeTask root = new OctreeTask(4, 0, 0, 0, -1, 0);
        queue.enqueueRoot(root);
        assertEquals(1, queue.queueSize(4), "L4 queue should have 1 task");
        for (int lvl = 0; lvl < 4; lvl++) {
            assertEquals(0, queue.queueSize(lvl), "L" + lvl + " should still be empty");
        }
    }

    @Test
    void enqueuedCountAt_tracksEnqueuedRoots() {
        OctreeQueue queue = new OctreeQueue();
        assertEquals(0, queue.enqueuedCountAt(4));

        queue.enqueueRoot(new OctreeTask(4, 0, 0, 0, -1, 0));
        assertEquals(1, queue.enqueuedCountAt(4));

        queue.enqueueRoot(new OctreeTask(4, 1, 0, 0, -1, 1));
        assertEquals(2, queue.enqueuedCountAt(4));
    }

    @Test
    void enqueuedCountAt_deduplicated_countsOnce() {
        OctreeQueue queue = new OctreeQueue();
        OctreeTask t = new OctreeTask(4, 0, 0, 0, -1, 0);
        queue.enqueueRoot(t);
        // Duplicate — should NOT increment enqueue count
        queue.enqueueRoot(new OctreeTask(4, 0, 0, 0, -1, 0));
        assertEquals(1, queue.enqueuedCountAt(4),
                "Duplicate enqueue should not increment counter");
    }

    @Test
    void completedCountAt_tracksCompletions() {
        OctreeQueue queue = new OctreeQueue();
        assertEquals(0, queue.completedCountAt(3));

        queue.markCompleted(3);
        assertEquals(1, queue.completedCountAt(3));
        queue.markCompleted(3);
        assertEquals(2, queue.completedCountAt(3));

        // Other levels unaffected
        assertEquals(0, queue.completedCountAt(0));
        assertEquals(0, queue.completedCountAt(4));
    }

    @Test
    void failedCountAt_tracksFailures() {
        OctreeQueue queue = new OctreeQueue();
        assertEquals(0, queue.failedCountAt(1));

        queue.markFailed(1);
        assertEquals(1, queue.failedCountAt(1));

        // Global counter also incremented
        assertEquals(1, queue.failedCount());
    }

    @Test
    void markCompleted_withLevel_updatesGlobalAndPerLevel() {
        OctreeQueue queue = new OctreeQueue();
        queue.markCompleted(2);
        queue.markCompleted(2);
        queue.markCompleted(0);

        assertEquals(3, queue.completedCount(), "Global count should be 3");
        assertEquals(2, queue.completedCountAt(2), "L2 per-level count should be 2");
        assertEquals(1, queue.completedCountAt(0), "L0 per-level count should be 1");
    }

    @Test
    void clear_resetsPerLevelCounters() {
        OctreeQueue queue = new OctreeQueue();
        queue.enqueueRoot(new OctreeTask(4, 0, 0, 0, -1, 0));
        queue.markCompleted(4);
        queue.markFailed(3);

        queue.clear();

        assertEquals(0, queue.enqueuedCountAt(4));
        assertEquals(0, queue.completedCountAt(4));
        assertEquals(0, queue.failedCountAt(3));
        assertEquals(0, queue.completedCount());
        assertEquals(0, queue.failedCount());
    }

    @Test
    void oldestPendingAgeMillis_emptyQueue_returnsZero() {
        OctreeQueue queue = new OctreeQueue();
        assertEquals(0L, queue.oldestPendingAgeMillis(4));
    }

    @Test
    void oldestPendingAgeMillis_withTask_returnsPositiveAge() throws InterruptedException {
        OctreeQueue queue = new OctreeQueue();
        queue.enqueueRoot(new OctreeTask(4, 0, 0, 0, -1, 0));
        Thread.sleep(5); // wait a few ms so age > 0
        long age = queue.oldestPendingAgeMillis(4);
        assertTrue(age >= 0L, "Oldest pending age should be non-negative");
    }

    // ── OctreeRuntimeStats snapshot ─────────────────────────────────────

    @Test
    void collect_nullQueue_returnsZeroStats() {
        int[] workers = {5, 3, 2, 1, 1};
        OctreeRuntimeStats stats = OctreeRuntimeStats.collect(
                null, workers, 10, 20, 32, true, System.currentTimeMillis() - 1000);

        assertNotNull(stats);
        assertEquals(0, stats.totalQueueSize());
        assertEquals(0, stats.totalCompleted());
        assertEquals(0, stats.totalFailed());
        assertTrue(stats.onnxActive());
        assertEquals(10, stats.playerSectionX());
        assertEquals(20, stats.playerSectionZ());
        assertEquals(32, stats.generationRadius());
    }

    @Test
    void collect_withQueue_capturesPerLevelData() {
        OctreeQueue queue = new OctreeQueue();
        // Enqueue 2 L4 roots
        queue.enqueueRoot(new OctreeTask(4, 0, 0, 0, -1, 0));
        queue.enqueueRoot(new OctreeTask(4, 1, 0, 0, -1, 1));
        // Mark 1 completed at L4
        queue.markCompleted(4);

        int[] workers = {5, 3, 2, 1, 1};
        long startMs = System.currentTimeMillis() - 5000; // 5 seconds ago
        OctreeRuntimeStats stats = OctreeRuntimeStats.collect(
                queue, workers, 0, 0, 32, true, startMs);

        OctreeRuntimeStats.PerLevelStats l4 = stats.levels()[4];
        assertEquals(4, l4.level());
        assertEquals(2, l4.queueSize(), "L4 should have 2 pending tasks");
        assertEquals(2, l4.totalEnqueued(), "L4 total enqueued should be 2");
        assertEquals(1, l4.totalCompleted(), "L4 total completed should be 1");
        assertEquals(1, l4.activeWorkers(), "L4 should have 1 worker");

        // L0 should be empty
        OctreeRuntimeStats.PerLevelStats l0 = stats.levels()[0];
        assertEquals(0, l0.queueSize());
        assertEquals(5, l0.activeWorkers(), "L0 should have 5 workers");

        assertEquals(2, stats.totalQueueSize(), "Total queue size should be 2");
        assertTrue(stats.uptimeMs() >= 4000L, "Uptime should be at least 4s");
    }

    @Test
    void perLevelStats_completionsPerSec_correctRate() {
        OctreeQueue queue = new OctreeQueue();
        queue.markCompleted(0);
        queue.markCompleted(0);  // 2 completions at L0

        int[] workers = {5, 3, 2, 1, 1};
        long startMs = System.currentTimeMillis() - 2000; // 2 seconds uptime
        OctreeRuntimeStats stats = OctreeRuntimeStats.collect(
                queue, workers, 0, 0, 32, true, startMs);

        double rate = stats.levels()[0].completionsPerSec(stats.uptimeMs());
        // 2 completions / ~2 seconds ≈ 1.0/sec
        assertTrue(rate > 0.5 && rate < 3.0,
                "Rate should be ~1.0/sec but was " + rate);
    }

    @Test
    void toDisplayString_containsRequiredFields() {
        int[] workers = {5, 3, 2, 1, 1};
        OctreeRuntimeStats stats = OctreeRuntimeStats.collect(
                null, workers, 12, 34, 32, true, System.currentTimeMillis() - 1000);

        String display = stats.toDisplayString();
        assertNotNull(display);
        assertTrue(display.contains("ONNX"), "Should show mode");
        assertTrue(display.contains("12"), "Should show player X");
        assertTrue(display.contains("queue sizes"), "Should show queue sizes row");
        assertTrue(display.contains("workers"), "Should show workers row");
        assertTrue(display.contains("L4"), "Should show L4");
        assertTrue(display.contains("L0"), "Should show L0");
    }

    @Test
    void toDisplayString_fallbackMode() {
        int[] workers = {5, 3, 2, 1, 1};
        OctreeRuntimeStats stats = OctreeRuntimeStats.collect(
                null, workers, 0, 0, 32, false, System.currentTimeMillis() - 1000);

        String display = stats.toDisplayString();
        assertTrue(display.contains("fallback"), "Should show fallback mode");
    }

    // ── PerformanceMonitor per-level timing ─────────────────────────────

    @Test
    void perLevelTiming_recordAndRetrieve() {
        PerformanceMonitor.addLevelTiming(2, 5_000_000L); // 5ms
        PerformanceMonitor.addLevelTiming(2, 3_000_000L); // 3ms

        double avg = PerformanceMonitor.getAverageLevelTiming(2);
        assertEquals(4.0, avg, 0.1, "Average should be ~4ms");
        assertEquals(2, PerformanceMonitor.getLevelTimingCount(2));
    }

    @Test
    void perLevelTiming_outOfRange_returnsZero() {
        assertEquals(0.0, PerformanceMonitor.getAverageLevelTiming(-1));
        assertEquals(0.0, PerformanceMonitor.getAverageLevelTiming(5));
        assertEquals(0L, PerformanceMonitor.getLevelTimingCount(-1));
    }

    @Test
    void perLevelTiming_resetClearsAllLevels() {
        PerformanceMonitor.addLevelTiming(0, 1_000_000L);
        PerformanceMonitor.addLevelTiming(4, 2_000_000L);

        PerformanceMonitor.reset();

        assertEquals(0.0, PerformanceMonitor.getAverageLevelTiming(0));
        assertEquals(0.0, PerformanceMonitor.getAverageLevelTiming(4));
    }

    @Test
    void levelTimingScope_recordsBothNamedAndPerLevel() throws Exception {
        try (var scope = PerformanceMonitor.startLevelTiming(3, "test_op")) {
            Thread.sleep(5); // ensure > 0ms
        }

        assertTrue(PerformanceMonitor.getAverageLevelTiming(3) > 0.0,
                "Per-level timing should be > 0");
        assertTrue(PerformanceMonitor.getAverageTiming("test_op") > 0.0,
                "Named timing should be > 0");
    }

    // ── enqueuedAtMs in OctreeTask ───────────────────────────────────────

    @Test
    void octreeTask_enqueuedAtMs_isRecentTimestamp() {
        long before = System.currentTimeMillis();
        OctreeTask task = new OctreeTask(4, 0, 0, 0, -1, 0);
        long after = System.currentTimeMillis();

        assertTrue(task.enqueuedAtMs >= before,
                "enqueuedAtMs should be >= creation start");
        assertTrue(task.enqueuedAtMs <= after,
                "enqueuedAtMs should be <= creation end");
    }
}
