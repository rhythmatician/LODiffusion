package com.rhythmatician.lodiffusion.voxy;

import java.util.function.IntFunction;

/**
 * Immutable snapshot of octree pipeline runtime metrics.
 *
 * <p>Collected on demand (e.g., for the {@code /lodiffusion octree} command)
 * without holding any lock.  Values are individually atomic but not
 * collectively consistent — treat them as approximate.
 *
 * @param levels          per-LOD stats, index 0 = L0 (finest) … 4 = L4 (coarsest)
 * @param playerSectionX  player section X at snapshot time (L0 coordinates)
 * @param playerSectionZ  player section Z at snapshot time (L0 coordinates)
 * @param generationRadius generation radius in sections (L0 coordinates)
 * @param onnxActive      {@code true} if ONNX pipeline is active; false for fallback
 * @param totalQueueSize  sum of all level queue sizes
 * @param totalCompleted  global completed-task count
 * @param totalFailed     global failed-task count
 * @param uptimeMs        elapsed ms since the pipeline started (for rate calculation)
 */
public record OctreeRuntimeStats(
        PerLevelStats[] levels,
        int playerSectionX,
        int playerSectionZ,
        int generationRadius,
        boolean onnxActive,
        int totalQueueSize,
        int totalCompleted,
        int totalFailed,
        long uptimeMs
) {

    /**
     * Per-LOD metrics snapshot.
     *
     * @param level           LOD level (0 = L0 finest … 4 = L4 coarsest)
     * @param queueSize       current number of pending tasks
     * @param oldestPendingMs approximate age (ms) of the oldest pending task
     * @param totalEnqueued   total tasks enqueued since pipeline start
     * @param totalCompleted  total tasks completed (ready) since pipeline start
     * @param totalFailed     total tasks failed since pipeline start
     * @param activeWorkers   number of worker threads assigned to this level
     * @param avgLatencyMs    average inference latency in ms (0 if no samples)
     */
    public record PerLevelStats(
            int level,
            int queueSize,
            long oldestPendingMs,
            long totalEnqueued,
            long totalCompleted,
            long totalFailed,
            int activeWorkers,
            double avgLatencyMs
    ) {

        /**
         * Jobs completed per second since pipeline start.
         *
         * @param uptimeMs pipeline uptime in milliseconds
         * @return rate, or 0 if uptime is 0
         */
        public double completionsPerSec(long uptimeMs) {
            if (uptimeMs <= 0) return 0.0;
            return totalCompleted * 1000.0 / uptimeMs;
        }

        /**
         * Jobs enqueued per second since pipeline start.
         *
         * @param uptimeMs pipeline uptime in milliseconds
         * @return rate, or 0 if uptime is 0
         */
        public double enqueuedPerSec(long uptimeMs) {
            if (uptimeMs <= 0) return 0.0;
            return totalEnqueued * 1000.0 / uptimeMs;
        }
    }

    /**
     * Collect a snapshot from a live {@link OctreeQueue} and supporting context.
     *
     * @param queue          the active octree queue (may be null if not started)
     * @param workerCounts   WORKERS_PER_LEVEL array (index 0=L0 … 4=L4)
     * @param playerSectionX player section X
     * @param playerSectionZ player section Z
     * @param generationRadius generation radius
     * @param onnxActive     whether ONNX pipeline is active
     * @param pipelineStartMs wall-clock ms when the pipeline started (for uptime)
     * @return a fresh snapshot
     */
    public static OctreeRuntimeStats collect(
            OctreeQueue queue,
            int[] workerCounts,
            int playerSectionX,
            int playerSectionZ,
            int generationRadius,
            boolean onnxActive,
            long pipelineStartMs) {

        long now = System.currentTimeMillis();
        long uptimeMs = (pipelineStartMs > 0) ? Math.max(1L, now - pipelineStartMs) : 0L;

        PerLevelStats[] levels = new PerLevelStats[OctreeQueue.NUM_LEVELS];
        int totalQ = 0;
        int totalCompleted = 0;
        int totalFailed = 0;

        for (int lvl = 0; lvl < OctreeQueue.NUM_LEVELS; lvl++) {
            int qs = 0;
            long oldest = 0L;
            long enqueued = 0L;
            long completed = 0L;
            long failed = 0L;
            int workers = (workerCounts != null && lvl < workerCounts.length)
                    ? workerCounts[lvl] : 0;

            if (queue != null) {
                qs        = queue.queueSize(lvl);
                oldest    = queue.oldestPendingAgeMillis(lvl);
                enqueued  = queue.enqueuedCountAt(lvl);
                completed = queue.completedCountAt(lvl);
                failed    = queue.failedCountAt(lvl);
            }

            double avgLatency = com.rhythmatician.lodiffusion.util.PerformanceMonitor
                    .getAverageLevelTiming(lvl);

            levels[lvl] = new PerLevelStats(lvl, qs, oldest, enqueued,
                    completed, failed, workers, avgLatency);
            totalQ += qs;
            totalCompleted += (int) completed;
            totalFailed    += (int) failed;
        }

        return new OctreeRuntimeStats(
                levels, playerSectionX, playerSectionZ,
                generationRadius, onnxActive,
                totalQ, totalCompleted, totalFailed, uptimeMs);
    }

    /**
     * Format the snapshot as human-readable text for in-game display.
     *
     * <p>Output example:
     * <pre>
     * mode: ONNX  player section: (12, -4)  radius: 32  uptime: 45s
     * queue sizes  L4:1  L3:5  L2:20  L1:80  L0:300
     * oldest age   L4:0ms  L3:200ms  L2:1s  L1:3s  L0:8s
     * enq/sec      L4:0.0  L3:0.1  L2:0.4  L1:1.2  L0:5.0
     * cmp/sec      L4:0.0  L3:0.1  L2:0.4  L1:1.2  L0:5.0
     * workers      L4:1  L3:1  L2:2  L1:3  L0:5
     * avg latency  L4:0ms  L3:0ms  L2:0ms  L1:0ms  L0:0ms
     * </pre>
     */
    public String toDisplayString() {
        StringBuilder sb = new StringBuilder();
        long upSec = uptimeMs / 1000;

        sb.append(String.format("mode: %s  player section: (%d, %d)  radius: %d  uptime: %ds%n",
                onnxActive ? "ONNX" : "fallback",
                playerSectionX, playerSectionZ, generationRadius, upSec));

        appendRow(sb, "queue sizes ", lvl -> String.valueOf(levels[lvl].queueSize()));
        appendRow(sb, "oldest age  ", lvl -> formatMs(levels[lvl].oldestPendingMs()));
        appendRow(sb, "enq/sec     ", lvl ->
                String.format("%.1f", levels[lvl].enqueuedPerSec(uptimeMs)));
        appendRow(sb, "cmp/sec     ", lvl ->
                String.format("%.1f", levels[lvl].completionsPerSec(uptimeMs)));
        appendRow(sb, "workers     ", lvl -> String.valueOf(levels[lvl].activeWorkers()));
        appendRow(sb, "avg latency ", lvl -> formatMs((long) levels[lvl].avgLatencyMs()));
        appendRow(sb, "total enq   ", lvl -> String.valueOf(levels[lvl].totalEnqueued()));
        appendRow(sb, "total cmp   ", lvl -> String.valueOf(levels[lvl].totalCompleted()));
        appendRow(sb, "total fail  ", lvl -> String.valueOf(levels[lvl].totalFailed()));

        return sb.toString();
    }

    /** Helper: append a labelled row with one value per level L4→L0. */
    private void appendRow(StringBuilder sb, String label,
                            java.util.function.IntFunction<String> valueFn) {
        sb.append(label);
        for (int lvl = 4; lvl >= 0; lvl--) {
            sb.append(String.format("  L%d:%s", lvl, valueFn.apply(lvl)));
        }
        sb.append('\n');
    }

    /** Format milliseconds as a compact human-readable string. */
    private static String formatMs(long ms) {
        if (ms <= 0) return "0ms";
        if (ms < 1000) return ms + "ms";
        if (ms < 60_000) return (ms / 1000) + "s";
        return (ms / 60_000) + "m" + ((ms % 60_000) / 1000) + "s";
    }
}
