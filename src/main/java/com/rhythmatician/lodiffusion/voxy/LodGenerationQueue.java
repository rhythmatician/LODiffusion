package com.rhythmatician.lodiffusion.voxy;

import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.PriorityBlockingQueue;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * Thread-safe priority queue system for the 4-stage ONNX inference pipeline.
 *
 * <p>Sections enter at stage 0 and are promoted through stages 1→2→3 as each
 * stage completes.  Each stage has its own {@link PriorityBlockingQueue}
 * ordered by distance from the player (closest first).
 *
 * <h3>Shutdown cascade</h3>
 * <p>Clean shutdown flows from upstream to downstream:
 * <ol>
 *   <li>Population finishes → {@link #signalPopulationDone()}</li>
 *   <li>All stage-0 workers drain their queue and exit →
 *       {@link #signalStageComplete(int) signalStageComplete(0)}</li>
 *   <li>Stage-1 worker drains its queue and exits →
 *       {@link #signalStageComplete(int) signalStageComplete(1)}</li>
 *   <li>… until stage 3 exits.</li>
 * </ol>
 *
 * <p>Each worker uses {@link #isUpstreamDone(int)} to decide when to exit
 * after its queue is empty.
 */
public final class LodGenerationQueue {

    static final int NUM_STAGES = 4;

    /** All tracked tasks, keyed by section position (deduplication). */
    private final ConcurrentHashMap<Long, SectionTask> allTasks =
            new ConcurrentHashMap<>();

    /** Per-stage priority queues.  Stage workers poll from their queue. */
    @SuppressWarnings({"unchecked", "rawtypes"})
    private final PriorityBlockingQueue<SectionTask>[] stageQueues =
            new PriorityBlockingQueue[NUM_STAGES];

    // ── Completion tracking ─────────────────────────────────────────────

    private final AtomicInteger completedCount = new AtomicInteger();
    private final AtomicInteger failedCount    = new AtomicInteger();
    private volatile int totalEnqueued;

    /** Set after all sections have been added to the stage-0 queue. */
    private final AtomicBoolean populationDone = new AtomicBoolean();

    /**
     * Per-stage completion flags.  Set when <em>all</em> workers for a
     * stage have exited (for stage 0, this means every thread in the pool).
     */
    private final AtomicBoolean[] stageComplete = new AtomicBoolean[NUM_STAGES];

    // ── Construction ────────────────────────────────────────────────────

    public LodGenerationQueue() {
        for (int i = 0; i < NUM_STAGES; i++) {
            stageQueues[i]  = new PriorityBlockingQueue<>();
            stageComplete[i] = new AtomicBoolean();
        }
    }

    // ── Enqueue / promote ───────────────────────────────────────────────

    /**
     * Enqueue a new task into the stage-0 queue.
     *
     * @return {@code true} if enqueued; {@code false} if a task with the
     *         same key already exists (duplicate)
     */
    public boolean enqueue(SectionTask task) {
        if (allTasks.putIfAbsent(task.key, task) != null) return false;
        stageQueues[0].add(task);
        return true;
    }

    /**
     * Promote a task to its next stage queue after successful processing.
     * The task's {@link SectionTask#nextStage()} must already have been
     * incremented via {@link SectionTask#promoteToNextStage(float[])}.
     */
    public void promoteToNextStage(SectionTask task) {
        int next = task.nextStage();
        if (next < NUM_STAGES) {
            stageQueues[next].add(task);
        }
    }

    // ── Polling ─────────────────────────────────────────────────────────

    /** Blocking poll with timeout. */
    public SectionTask poll(int stage, long timeout, TimeUnit unit)
            throws InterruptedException {
        return stageQueues[stage].poll(timeout, unit);
    }

    /** Non-blocking poll. */
    public SectionTask poll(int stage) {
        return stageQueues[stage].poll();
    }

    // ── Completion signals ──────────────────────────────────────────────

    /** Signal that all sections have been added to the stage-0 queue. */
    public void signalPopulationDone() { populationDone.set(true); }

    /**
     * Signal that all workers for a stage have exited.
     * Downstream workers use this to know when their queue is permanently
     * drained.
     */
    public void signalStageComplete(int stage) { stageComplete[stage].set(true); }

    /** Increment the completed-sections counter (called after Voxy write). */
    public void markCompleted() { completedCount.incrementAndGet(); }

    /** Increment the failed-sections counter. */
    public void markFailed() { failedCount.incrementAndGet(); }

    /**
     * Check whether a stage worker should exit because no more tasks will
     * arrive in its queue.
     * <ul>
     *   <li>Stage 0: exits when population is done and queue is empty.</li>
     *   <li>Stage N&gt;0: exits when stage N−1 is complete and queue is empty.</li>
     * </ul>
     *
     * <p><b>Thread safety:</b> the upstream-done flag is set <em>after</em>
     * all upstream queue additions, so checking the flag then the queue size
     * is eventually consistent.  Callers should do a final non-blocking poll
     * after seeing upstream-done to avoid TOCTOU misses.
     */
    public boolean isUpstreamDone(int stage) {
        if (stage == 0) return populationDone.get();
        return stageComplete[stage - 1].get();
    }

    // ── Accessors ───────────────────────────────────────────────────────

    public void setTotalEnqueued(int total) { this.totalEnqueued = total; }
    public int totalEnqueued()  { return totalEnqueued; }
    public int completedCount() { return completedCount.get(); }
    public int failedCount()    { return failedCount.get(); }
    public int stageQueueSize(int stage) { return stageQueues[stage].size(); }

    /** Clear all state for reuse or shutdown. */
    public void clear() {
        for (int i = 0; i < NUM_STAGES; i++) {
            stageQueues[i].clear();
            stageComplete[i].set(false);
        }
        allTasks.clear();
        completedCount.set(0);
        failedCount.set(0);
        totalEnqueued = 0;
        populationDone.set(false);
    }
}
