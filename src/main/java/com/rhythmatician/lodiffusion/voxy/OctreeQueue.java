package com.rhythmatician.lodiffusion.voxy;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;
import java.util.function.BiFunction;

import com.rhythmatician.lodiffusion.HelloTerrainMod;

/**
 * Priority queue system for octree-based LOD generation.
 *
 * <p>5 queues (one per LOD level 0-4), processed L4-first for breadth-first
 * tree traversal.  After inference at level N, occupied octants spawn child
 * tasks at level N-1.
 *
 * <h3>Threading model</h3>
 * <p>Multiple worker threads may drain different levels concurrently.
 * The queue provides thread-safe enqueue, drain, and child-spawning.
 *
 * <h3>Shutdown</h3>
 * <p>Shutdown cascades top-down: when L4 is done generating, its completion
 * is signalled, and workers at L3 know no more parents will arrive, etc.
 * Each worker uses {@link #isUpstreamDone(int)} to decide when to exit.
 *
 * @see OctreeTask
 */
public final class OctreeQueue {

    /** Number of LOD levels (0-4 inclusive). */
    static final int NUM_LEVELS = 5;

    /** All tracked tasks, keyed by packed wsKey (deduplication). */
    private final ConcurrentHashMap<Long, OctreeTask> allTasks =
            new ConcurrentHashMap<>();

    /** Per-level priority queues.  Index 0 = L0 (finest), 4 = L4 (coarsest). */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private final PriorityBlockingQueue<OctreeTask>[] levelQueues =
            new PriorityBlockingQueue[NUM_LEVELS];

    // ── Completion tracking ─────────────────────────────────────────────

    private final AtomicInteger completedCount = new AtomicInteger();
    private final AtomicInteger failedCount    = new AtomicInteger();
    private volatile int totalEnqueued;

    /**
     * Set after all L4 root tasks have been enqueued (population phase done).
     * L4 workers check this + queue empty to know when to exit.
     */
    private final AtomicBoolean populationDone = new AtomicBoolean();

    /**
     * Per-level completion flags.  Set when all workers for a level have
     * exited.  Used by child levels to know when no more parent spawns
     * will arrive.
     */
    private final AtomicBoolean[] levelComplete = new AtomicBoolean[NUM_LEVELS];

    /**
     * Lock for re-prioritisation.  Held during drain–update–re-add cycles
     * to prevent workers from seeing transiently empty queues.
     */
    private final ReentrantLock reprioritiseLock = new ReentrantLock();

    /**
     * Callback for building column context for child tasks.
     * {@code (childLevel, childTask) → OctreeColumnContext}.
     * Set via {@link #setColumnContextBuilder}.
     */
    private volatile BiFunction<Integer, OctreeTask, OctreeColumnContext> columnContextBuilder;

    // ── Octree efficiency stats (RocNet-inspired) ─────────────────────
    // Track how many octants are spawned vs pruned at each level to
    // measure the efficiency advantage the octree is supposed to buy.

    /** Per-level count of octants spawned (occupied). */
    private final AtomicInteger[] spawnedPerLevel = new AtomicInteger[NUM_LEVELS];

    /** Per-level count of octants pruned (empty). */
    private final AtomicInteger[] prunedPerLevel = new AtomicInteger[NUM_LEVELS];

    // ── Construction ────────────────────────────────────────────────────

    public OctreeQueue() {
        for (int i = 0; i < NUM_LEVELS; i++) {
            levelQueues[i]  = new PriorityBlockingQueue<>();
            levelComplete[i] = new AtomicBoolean();
            spawnedPerLevel[i] = new AtomicInteger();
            prunedPerLevel[i]  = new AtomicInteger();
        }
    }

    /**
     * Set the callback used by {@link #spawnChildren} to build column
     * context for child tasks.  Must be called before the pipeline starts.
     */
    public void setColumnContextBuilder(
            BiFunction<Integer, OctreeTask, OctreeColumnContext> builder) {
        this.columnContextBuilder = builder;
    }

    // ── Enqueue ─────────────────────────────────────────────────────────

    /**
     * Enqueue a root L4 task.
     *
     * @return {@code true} if enqueued; {@code false} if a task with the
     *         same key already exists (duplicate)
     */
    public boolean enqueueRoot(OctreeTask task) {
        if (task.level != 4) {
            throw new IllegalArgumentException(
                    "enqueueRoot: expected L4 task, got L" + task.level);
        }
        if (allTasks.putIfAbsent(task.wsKey, task) != null) return false;
        levelQueues[4].add(task);
        return true;
    }

    /**
     * Enqueue a child task at any level (internal use by spawnChildren).
     *
     * @return {@code true} if enqueued; {@code false} if duplicate
     */
    boolean enqueueChild(OctreeTask task) {
        if (allTasks.putIfAbsent(task.wsKey, task) != null) return false;
        levelQueues[task.level].add(task);
        return true;
    }

    // ── Child spawning ──────────────────────────────────────────────────

    /**
     * After inference on a parent task, spawn child tasks for each occupied
     * octant.  This is the core of the octree traversal.
     *
     * <p>For each set bit in {@code occMask}:
     * <ol>
     *   <li>Compute child WorldSection coordinates from parent + octant</li>
     *   <li>Extract the 16³ octant region from the parent's 32³ argmax
     *       predictions</li>
     *   <li>Upsample 16³ → 32³ via nearest-neighbor to produce the child's
     *       parent context</li>
     *   <li>Build column context for the child's footprint</li>
     *   <li>Create and enqueue the child {@link OctreeTask}</li>
     * </ol>
     *
     * <p>Does nothing if the parent is at L0 (leaves have no children).
     *
     * @param parent      the parent task that just completed inference
     * @param occMask     8-bit occupancy mask from sigmoid(occ_logits) > 0.5
     * @param blockArgmax the parent's 32³ argmax block IDs as
     *                    {@code float[32][32][32]} (Y, Z, X order),
     *                    already converted from logits
     * @return number of children actually enqueued (may be less than
     *         popcount(occMask) if duplicates were detected)
     */
    public int spawnChildren(OctreeTask parent, byte occMask,
                             float[][][] blockArgmax) {
        if (parent.level <= 0) return 0;

        int childLevel = parent.level - 1;
        int spawned = 0;
        int pruned = 0;

        for (int oct = 0; oct < 8; oct++) {
            if ((occMask & (1 << oct)) == 0) {
                pruned++;
                continue;
            }

            int cx = OctreeTask.childX(parent.wsX, oct);
            int cy = OctreeTask.childY(parent.wsY, oct);
            int cz = OctreeTask.childZ(parent.wsZ, oct);

            // Extract the 16³ octant from the parent's 32³ predictions
            // Octant bits: bit0=X, bit1=Z, bit2=Y
            int offX = (oct & 1) * 16;
            int offY = ((oct >> 2) & 1) * 16;
            int offZ = ((oct >> 1) & 1) * 16;

            // Upsample 16³ → 32³ via nearest-neighbor, flatten to
            // long[32 * 32 * 32] for the ONNX parent_blocks input
            long[] childParentFlat = extractAndUpsampleOctant(
                    blockArgmax, offY, offZ, offX);

            OctreeTask child = new OctreeTask(
                    childLevel, cx, cy, cz, oct, parent.priority);
            child.parentContextFlat = childParentFlat;

            // Build column context for the child's geographic footprint
            if (columnContextBuilder != null) {
                child.columnContext = columnContextBuilder.apply(childLevel, child);
            }

            if (enqueueChild(child)) {
                spawned++;
            }
        }

        if (spawned > 0 || pruned > 0) {
            spawnedPerLevel[childLevel].addAndGet(spawned);
            prunedPerLevel[childLevel].addAndGet(pruned);
        }

        if (spawned > 0) {
            HelloTerrainMod.LOGGER.debug(
                    "[OctreeQueue] Spawned {} children at L{} from parent L{} ({},{},{}) — pruned {}",
                    spawned, childLevel, parent.level,
                    parent.wsX, parent.wsY, parent.wsZ, pruned);
        }

        return spawned;
    }

    /**
     * Extract a 16³ octant from a 32³ volume and upsample 2× via
     * nearest-neighbor to produce a flat 32³ array of block IDs.
     *
     * <p>The result is shaped as {@code long[32 * 32 * 32]} in
     * row-major Y,Z,X order matching the ONNX parent_blocks input
     * layout {@code [N, 32, 32, 32]}.
     *
     * @param src  source 32³ argmax values, indexed [Y][Z][X]
     * @param offY Y offset of the octant (0 or 16)
     * @param offZ Z offset of the octant (0 or 16)
     * @param offX X offset of the octant (0 or 16)
     * @return flat long[32768] containing the upsampled octant block IDs
     */
    static long[] extractAndUpsampleOctant(float[][][] src,
                                            int offY, int offZ, int offX) {
        long[] dst = new long[32 * 32 * 32];
        int idx = 0;
        for (int dy = 0; dy < 32; dy++) {
            int srcY = offY + (dy >> 1);  // nearest-neighbor: /2
            for (int dz = 0; dz < 32; dz++) {
                int srcZ = offZ + (dz >> 1);
                for (int dx = 0; dx < 32; dx++) {
                    int srcX = offX + (dx >> 1);
                    dst[idx++] = (long) src[srcY][srcZ][srcX];
                }
            }
        }
        return dst;
    }

    // ── Polling ─────────────────────────────────────────────────────────

    /**
     * Poll the next task, processing higher (coarser) levels first to
     * achieve breadth-first traversal.  Scans from L4 down to L0.
     *
     * @param timeout maximum time to wait for a task
     * @param unit    time unit for the timeout
     * @return the next highest-priority task from the coarsest non-empty
     *         level, or {@code null} if the timeout expired
     */
    public OctreeTask pollNext(long timeout, TimeUnit unit)
            throws InterruptedException {
        // Try each level from coarsest to finest (breadth-first)
        for (int lvl = 4; lvl >= 0; lvl--) {
            OctreeTask task = levelQueues[lvl].poll();
            if (task != null) return task;
        }
        // Nothing immediately available — wait briefly on the coarsest
        // non-empty-or-active level
        for (int lvl = 4; lvl >= 0; lvl--) {
            if (!isLevelPermanentlyDone(lvl)) {
                return levelQueues[lvl].poll(timeout, unit);
            }
        }
        return null;
    }

    /**
     * Drain up to {@code maxBatch} tasks from a specific level's queue.
     *
     * <p>Performs a blocking poll with timeout for the first task, then
     * greedily drains additional tasks without blocking.  This enables
     * batched inference per level.
     *
     * @param level    LOD level (0-4)
     * @param maxBatch maximum tasks to return
     * @param timeout  how long to wait for the first task
     * @param unit     time unit
     * @return list of 0 to maxBatch tasks
     */
    public List<OctreeTask> drainLevel(int level, int maxBatch,
                                        long timeout, TimeUnit unit)
            throws InterruptedException {
        List<OctreeTask> batch = new ArrayList<>(maxBatch);

        OctreeTask first = levelQueues[level].poll(timeout, unit);
        if (first == null) return batch;
        batch.add(first);

        while (batch.size() < maxBatch) {
            OctreeTask next = levelQueues[level].poll();
            if (next == null) break;
            batch.add(next);
        }
        return batch;
    }

    /** Non-blocking poll from a specific level. */
    public OctreeTask pollLevel(int level) {
        return levelQueues[level].poll();
    }

    // ── Completion signals ──────────────────────────────────────────────

    /** Signal that all L4 root tasks have been enqueued. */
    public void signalPopulationDone() { populationDone.set(true); }

    /**
     * Signal that all workers for a level have exited.
     * Child-level workers use this to know when their queue is permanently
     * drained (no more parents will spawn children).
     */
    public void signalLevelComplete(int level) { levelComplete[level].set(true); }

    /** Increment the completed-task counter (called after Voxy write). */
    public void markCompleted() { completedCount.incrementAndGet(); }

    /** Increment the failed-task counter. */
    public void markFailed() { failedCount.incrementAndGet(); }

    /**
     * Check whether a level worker should exit because no more tasks
     * will arrive in its queue.
     *
     * <ul>
     *   <li>L4: exits when population is done and queue is empty.</li>
     *   <li>L3-L0: exits when the parent level (level+1) is complete
     *       and its queue is empty.</li>
     * </ul>
     */
    public boolean isUpstreamDone(int level) {
        if (level == 4) return populationDone.get();
        return levelComplete[level + 1].get();
    }

    /** Check if a level is permanently done (upstream done + queue empty). */
    private boolean isLevelPermanentlyDone(int level) {
        return isUpstreamDone(level) && levelQueues[level].isEmpty();
    }

    /**
     * Atomically check exit conditions and drain remaining tasks for a level.
     *
     * <p>Holds the {@code reprioritiseLock} during the check-and-drain to
     * prevent a race where {@link #reprioritise} / {@link #reprioritiseDirectional}
     * drains the queue (making it appear empty) while a worker decides to exit.
     *
     * <p>Workers <em>must</em> use this method instead of separately calling
     * {@link #isUpstreamDone(int)} and {@link #pollLevel(int)} to avoid
     * premature termination when a reprioritisation is in progress.
     *
     * @param level LOD level (0-4)
     * @return remaining tasks drained from the queue if upstream is done and
     *         the queue is empty after draining; {@code null} if the level is
     *         not yet eligible to exit (upstream not done, or tasks remain)
     */
    public List<OctreeTask> tryFinalDrain(int level) {
        reprioritiseLock.lock();
        try {
            if (!isUpstreamDone(level)) return null;
            List<OctreeTask> remaining = new ArrayList<>();
            levelQueues[level].drainTo(remaining);
            return remaining;
        } finally {
            reprioritiseLock.unlock();
        }
    }

    // ── Live re-prioritisation ──────────────────────────────────────────

    /**
     * Re-heap all level queues using updated priorities based on the
     * player's current section position (L0 coordinates).
     */
    public void reprioritise(int playerSectionX, int playerSectionZ) {
        reprioritiseLock.lock();
        try {
            for (int lvl = 0; lvl < NUM_LEVELS; lvl++) {
                List<OctreeTask> tmp = new ArrayList<>();
                levelQueues[lvl].drainTo(tmp);
                for (OctreeTask t : tmp) {
                    if (!t.isCancelled()) {
                        t.updatePriority(playerSectionX, playerSectionZ);
                    }
                }
                levelQueues[lvl].addAll(tmp);
            }
        } finally {
            reprioritiseLock.unlock();
        }
    }

    /**
     * Re-heap using direction-weighted priorities.
     */
    public void reprioritiseDirectional(int playerSectionX, int playerSectionZ,
                                         float headingX, float headingZ,
                                         float coneStrength) {
        reprioritiseLock.lock();
        try {
            for (int lvl = 0; lvl < NUM_LEVELS; lvl++) {
                List<OctreeTask> tmp = new ArrayList<>();
                levelQueues[lvl].drainTo(tmp);
                for (OctreeTask t : tmp) {
                    if (!t.isCancelled()) {
                        t.updateDirectionalPriority(playerSectionX, playerSectionZ,
                                headingX, headingZ, coneStrength);
                    }
                }
                levelQueues[lvl].addAll(tmp);
            }
        } finally {
            reprioritiseLock.unlock();
        }
    }

    /**
     * Cancel all PENDING tasks whose Manhattan distance from the given
     * centre exceeds {@code maxRadius}.  Distance is computed at each
     * task's native level.
     *
     * @return number of tasks cancelled
     */
    public int cancelBeyondRadius(int playerSectionX, int playerSectionZ,
                                   int maxRadius) {
        int cancelled = 0;
        for (Map.Entry<Long, OctreeTask> entry : allTasks.entrySet()) {
            OctreeTask t = entry.getValue();
            int playerAtLevel_X = playerSectionX >> t.level;
            int playerAtLevel_Z = playerSectionZ >> t.level;
            int dist = Math.abs(t.wsX - playerAtLevel_X)
                     + Math.abs(t.wsZ - playerAtLevel_Z);
            if (dist > (maxRadius >> t.level) && t.cancel()) {
                cancelled++;
            }
        }
        // Clean up terminal tasks from the dedup map
        allTasks.entrySet().removeIf(e -> {
            OctreeTask.State s = e.getValue().state();
            return s == OctreeTask.State.CANCELLED
                || s == OctreeTask.State.READY
                || s == OctreeTask.State.FAILED;
        });
        return cancelled;
    }

    /**
     * Remove a task from the dedup map so the same section can be
     * re-enqueued later.
     */
    public void removeFromDedup(long key) {
        allTasks.remove(key);
    }

    // ── Stats / Accessors ───────────────────────────────────────────────

    public void setTotalEnqueued(int total) { this.totalEnqueued = total; }
    public void addTotalEnqueued(int delta) { this.totalEnqueued += delta; }
    public int totalEnqueued()  { return totalEnqueued; }
    public int completedCount() { return completedCount.get(); }
    public int failedCount()    { return failedCount.get(); }
    public int totalProcessed() { return completedCount.get() + failedCount.get(); }
    public int levelQueueSize(int level) { return levelQueues[level].size(); }
    public int trackedTaskCount() { return allTasks.size(); }

    /**
     * Format queue sizes as a compact string for logging:
     * {@code "L4:12|L3:45|L2:100|L1:230|L0:500"}.
     */
    public String queueSizeSummary() {
        StringBuilder sb = new StringBuilder();
        for (int lvl = 4; lvl >= 0; lvl--) {
            if (sb.length() > 0) sb.append('|');
            sb.append("L").append(lvl).append(':').append(levelQueues[lvl].size());
        }
        return sb.toString();
    }

    /**
     * Format octree efficiency stats as a compact string for logging.
     *
     * <p>RocNet-inspired: this shows how effectively the octree prunes
     * empty subtrees at each level.  A well-calibrated model should
     * prune most octants at coarse levels and fewer at fine levels.
     *
     * <p>Example: {@code "L3: 120 spawned / 200 pruned (62.5% pruned) |
     * L2: 480 spawned / 480 pruned (50.0% pruned)"}.
     *
     * @return formatted efficiency summary, or empty string if no stats
     */
    public String efficiencySummary() {
        StringBuilder sb = new StringBuilder();
        for (int lvl = 3; lvl >= 0; lvl--) {
            int s = spawnedPerLevel[lvl].get();
            int p = prunedPerLevel[lvl].get();
            int total = s + p;
            if (total == 0) continue;
            if (sb.length() > 0) sb.append(" | ");
            double prunePct = 100.0 * p / total;
            sb.append(String.format("L%d: %d spawned / %d pruned (%.1f%% pruned)",
                    lvl, s, p, prunePct));
        }
        return sb.toString();
    }

    /** Per-level spawn count accessor. */
    public int spawnedAt(int level) { return spawnedPerLevel[level].get(); }

    /** Per-level prune count accessor. */
    public int prunedAt(int level) { return prunedPerLevel[level].get(); }

    /** Clear all state for reuse or shutdown. */
    public void clear() {
        for (int i = 0; i < NUM_LEVELS; i++) {
            levelQueues[i].clear();
            levelComplete[i].set(false);
            spawnedPerLevel[i].set(0);
            prunedPerLevel[i].set(0);
        }
        allTasks.clear();
        completedCount.set(0);
        failedCount.set(0);
        totalEnqueued = 0;
        populationDone.set(false);
    }
}
