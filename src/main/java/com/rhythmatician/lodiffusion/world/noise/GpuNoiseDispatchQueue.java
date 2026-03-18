package com.rhythmatician.lodiffusion.world.noise;

import io.github.lodiffusion.worldgen.QuartNoiseCompute;
import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentLinkedQueue;
import java.util.concurrent.atomic.AtomicLong;

/**
 * Cross-thread dispatch queue that bridges the LOD generation thread
 * (which has no GL context) to the render thread (which owns the GL context
 * and can execute {@link QuartNoiseCompute} dispatches).
 *
 * <h2>Threading model</h2>
 * <ul>
 *   <li><b>Gen thread</b> ({@code LODiffusion-Gen}) calls {@link #enqueue(int, int, int)},
 *       which returns a {@link CompletableFuture}&lt;{@link SectionNoiseData}&gt;.
 *       The gen thread blocks on {@code future.get(timeout)} (see
 *       {@link GpuNoiseRouterSampler}).</li>
 *   <li><b>Render thread</b> calls {@link #tickDrain()} once per client tick
 *       (via {@code ClientTickEvents.END_CLIENT_TICK}).  This pops up to
 *       {@link #MAX_DRAIN_PER_TICK} requests, batches them into a single GPU
 *       dispatch via {@link QuartNoiseCompute#compute(int[][], int)}, and
 *       completes the corresponding futures with the results.</li>
 * </ul>
 *
 * <h2>Lifecycle</h2>
 * <pre>
 *   WorldGenEventHandler.onWorldLoad  → GpuNoiseDispatchQueue.init(quartCompute)
 *   LodiffusionClient END_CLIENT_TICK → GpuNoiseDispatchQueue.tickDrain()
 *   WorldGenEventHandler.onWorldUnload→ GpuNoiseDispatchQueue.shutdown()
 * </pre>
 *
 * @see GpuNoiseRouterSampler
 * @see QuartNoiseCompute
 */
public final class GpuNoiseDispatchQueue {
    private static final Logger LOGGER = LogManager.getLogger();

    /** Maximum requests drained per client tick (20 ticks/sec → 640 sections/sec). */
    public static final int MAX_DRAIN_PER_TICK = 32;

    /** Volatile singleton — set by {@link #init}, cleared by {@link #shutdown}. */
    private static volatile GpuNoiseDispatchQueue INSTANCE;

    // ── Instance state ──────────────────────────────────────────────────

    private final QuartNoiseCompute compute;
    private final ConcurrentLinkedQueue<NoiseRequest> queue = new ConcurrentLinkedQueue<>();

    // Metrics (thread-safe counters)
    private final AtomicLong totalEnqueued = new AtomicLong();
    private final AtomicLong totalDispatched = new AtomicLong();
    private final AtomicLong totalFailed = new AtomicLong();

    // ── Inner record ────────────────────────────────────────────────────

    /**
     * A pending GPU noise request with its completion handle.
     */
    record NoiseRequest(int sectionX, int sectionY, int sectionZ,
                        CompletableFuture<SectionNoiseData> future) {
    }

    // ── Constructor (private — use init()) ──────────────────────────────

    private GpuNoiseDispatchQueue(QuartNoiseCompute compute) {
        this.compute = compute;
    }

    // ── Static lifecycle ────────────────────────────────────────────────

    /**
     * Initialise the singleton dispatch queue.
     * Called on the render thread during world load (after QuartNoiseCompute is ready).
     *
     * @param quartCompute the initialised QuartNoiseCompute instance
     */
    public static void init(QuartNoiseCompute quartCompute) {
        if (quartCompute == null) {
            LOGGER.warn("[GpuNoiseDispatchQueue] init called with null QuartNoiseCompute — skipping");
            return;
        }
        if (!quartCompute.isReady()) {
            LOGGER.warn("[GpuNoiseDispatchQueue] init called but QuartNoiseCompute is not ready — skipping");
            return;
        }
        GpuNoiseDispatchQueue old = INSTANCE;
        if (old != null) {
            LOGGER.warn("[GpuNoiseDispatchQueue] Replacing existing instance — draining {} pending requests",
                    old.queue.size());
            old.cancelAll("Queue replaced by new world load");
        }
        INSTANCE = new GpuNoiseDispatchQueue(quartCompute);
        LOGGER.info("[GpuNoiseDispatchQueue] Initialised (maxDrainPerTick={})", MAX_DRAIN_PER_TICK);
    }

    /**
     * Shut down the singleton dispatch queue.
     * Called on the render thread during world unload.  Any pending futures are
     * completed exceptionally so blocked gen threads unblock immediately.
     */
    public static void shutdown() {
        GpuNoiseDispatchQueue q = INSTANCE;
        INSTANCE = null;
        if (q != null) {
            int remaining = q.queue.size();
            q.cancelAll("World unloading");
            LOGGER.info("[GpuNoiseDispatchQueue] Shutdown — cancelled {} pending, " +
                            "stats: enqueued={}, dispatched={}, failed={}",
                    remaining, q.totalEnqueued.get(), q.totalDispatched.get(), q.totalFailed.get());
        }
    }

    /**
     * Returns the current singleton instance, or {@code null} if not initialised.
     */
    public static GpuNoiseDispatchQueue instance() {
        return INSTANCE;
    }

    // ── Public API (gen thread) ─────────────────────────────────────────

    /**
     * Enqueue a section for GPU noise evaluation.
     * Called from the gen thread.  The returned future will be completed on the
     * render thread when the GPU dispatch finishes.
     *
     * @param sectionX chunk-X coordinate
     * @param sectionY section-Y coordinate
     * @param sectionZ chunk-Z coordinate
     * @return future that resolves to the sampled noise data
     */
    public CompletableFuture<SectionNoiseData> enqueue(int sectionX, int sectionY, int sectionZ) {
        CompletableFuture<SectionNoiseData> future = new CompletableFuture<>();
        queue.add(new NoiseRequest(sectionX, sectionY, sectionZ, future));
        totalEnqueued.incrementAndGet();
        return future;
    }

    // ── Public API (render thread) ──────────────────────────────────────

    /**
     * Drain up to {@link #MAX_DRAIN_PER_TICK} requests and dispatch them as a
     * single GPU batch.  Must be called on the render thread (GL context required).
     *
     * <p>This is the only method that touches {@link QuartNoiseCompute}, ensuring
     * all GL calls happen on the render thread.
     */
    public void drainAndDispatch() {
        drainAndDispatch(MAX_DRAIN_PER_TICK);
    }

    /**
     * Drain up to {@code maxBatch} requests and dispatch them.
     *
     * @param maxBatch maximum number of requests to drain this tick
     */
    public void drainAndDispatch(int maxBatch) {
        if (queue.isEmpty()) return;

        // Collect up to maxBatch requests
        NoiseRequest[] batch = new NoiseRequest[maxBatch];
        int count = 0;
        for (int i = 0; i < maxBatch; i++) {
            NoiseRequest req = queue.poll();
            if (req == null) break;
            batch[count++] = req;
        }
        if (count == 0) return;

        try {
            // Build section origins in block coordinates (QuartNoiseCompute expects blockX/Y/Z)
            int[][] origins = new int[count][3];
            for (int i = 0; i < count; i++) {
                origins[i][0] = batch[i].sectionX() * 16;
                origins[i][1] = batch[i].sectionY() * 16;
                origins[i][2] = batch[i].sectionZ() * 16;
            }

            // GPU dispatch + readback (all on render thread)
            SectionNoiseData[] results = compute.compute(origins, count);

            // Complete futures with results
            for (int i = 0; i < count; i++) {
                batch[i].future().complete(results[i]);
            }
            totalDispatched.addAndGet(count);

        } catch (Exception e) {
            // Complete all futures exceptionally so gen thread doesn't hang
            LOGGER.error("[GpuNoiseDispatchQueue] GPU dispatch failed for batch of {} — " +
                    "completing futures exceptionally", count, e);
            for (int i = 0; i < count; i++) {
                batch[i].future().completeExceptionally(e);
            }
            totalFailed.addAndGet(count);
        }
    }

    // ── Static convenience for tick hook ────────────────────────────────

    /**
     * Called from {@code ClientTickEvents.END_CLIENT_TICK}.
     * No-ops if the queue is not initialised.
     */
    public static void tickDrain() {
        GpuNoiseDispatchQueue q = INSTANCE;
        if (q != null) {
            q.drainAndDispatch();
        }
    }

    // ── Internal helpers ────────────────────────────────────────────────

    /**
     * Cancel all pending requests by completing their futures exceptionally.
     */
    private void cancelAll(String reason) {
        NoiseRequest req;
        int cancelled = 0;
        while ((req = queue.poll()) != null) {
            req.future().completeExceptionally(
                    new IllegalStateException("GPU dispatch queue cancelled: " + reason));
            cancelled++;
        }
        if (cancelled > 0) {
            LOGGER.debug("[GpuNoiseDispatchQueue] Cancelled {} pending requests: {}", cancelled, reason);
        }
    }

    // ── Metrics accessors ───────────────────────────────────────────────

    public long getTotalEnqueued()   { return totalEnqueued.get(); }
    public long getTotalDispatched() { return totalDispatched.get(); }
    public long getTotalFailed()     { return totalFailed.get(); }
    public int  getPendingCount()    { return queue.size(); }
}
