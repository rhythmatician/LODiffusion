package com.rhythmatician.lodiffusion.voxy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicInteger;

import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.HelloTerrainMod;
import com.rhythmatician.lodiffusion.onnx.ModelConfig;
import com.rhythmatician.lodiffusion.onnx.BlockVocabulary;
import com.rhythmatician.lodiffusion.onnx.InferenceResult;
import com.rhythmatician.lodiffusion.onnx.ProgressiveModelRunner;

import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkStatus;

/**
 * Background service that generates terrain around the player and pushes it
 * into Voxy for distant rendering.
 *
 * <h3>Architecture — Per-Stage Pipeline</h3>
 * <p>The ONNX model is a 4-stage <em>refinement pipeline</em>.  Each section
 * flows through all 4 stages, closest-to-player first:
 *
 * <ol>
 *   <li><b>Stage 0</b> (init → LOD4) — a pool of workers processes sections
 *       with no parent dependency.</li>
 *   <li><b>Stage 1</b> (LOD4 → LOD3) — single worker thread, takes the
 *       binary air mask from stage 0 as parent input.</li>
 *   <li><b>Stage 2</b> (LOD3 → LOD2) — single worker thread.</li>
 *   <li><b>Stage 3</b> (LOD2 → LOD1) — single worker thread, produces the
 *       final 16³ output and writes it to Voxy.</li>
 * </ol>
 *
 * <p>Sections are prioritized by Manhattan distance from the player so
 * nearby terrain appears first.  The pipeline ensures maximum throughput
 * by keeping all stages busy concurrently.
 */
public final class LodGenerationService {

    /** Model LOD token range. */
    private static final int COARSEST_LOD = 4;

    /**
     * Radius (in sections) for each LOD pass.  Index = LOD level.
     * LOD 4 covers the most area (coarsest, cheapest), LOD 1 the least.
     */
    private static final int[] PASS_RADIUS = {0, 4, 8, 12, 16};

    /** How many sections of Y range to generate (from y=-64 upward). */
    private static final int Y_SECTIONS = 16;  // y sections -4..11 → blocks -64..191
    private static final int Y_BASE_SECTION = -4;  // start at y=-64

    /**
     * Extra margin (in sections) above and below the surface to generate.
     * Ensures caves near the surface, tree canopies, and hilly terrain are
     * captured.  Sections outside surface ± margin are skipped entirely.
     */
    private static final int SURFACE_MARGIN = 1;  // 1 section = 16 blocks

    /**
     * When true, force the air mask to predict air for voxels above the
     * surface heightmap.  This compensates for undertrained models that
     * predict all-solid, preventing terrain from extending above the
     * real surface.  Can be disabled once the model learns air boundaries.
     */
    private static final boolean HEIGHTMAP_CLIP = true;

    /**
     * Number of parallel worker threads for stage 0 (init → LOD4).
     * Stage 0 has no parent dependency so sections can run concurrently.
     * Capped at 4 to avoid starving the game thread.
     */
    private static final int STAGE_0_PARALLELISM =
            Math.min(Runtime.getRuntime().availableProcessors(), 4);

    /**
     * Maximum number of sections to batch into a single ONNX inference call.
     * Dynamic-batch ONNX models amortize per-call overhead across the batch,
     * improving throughput significantly.  Empirically, 8–16 gives a good
     * balance between throughput and latency.
     *
     * <p>Set to 1 to disable batching (falls back to single-sample mode).
     */
    private static final int MAX_BATCH_SIZE = 8;

    /**
     * Generation radius (in sections).  All sections within this Manhattan
     * distance from the player are generated, closest first.
     */
    private static final int GENERATION_RADIUS =
            Config.getInt("generationRadius", PASS_RADIUS[COARSEST_LOD]);

    /**
     * Extra margin (in sections) beyond GENERATION_RADIUS before tasks
     * are cancelled.  Prevents thrashing when the player oscillates
     * near the boundary.
     */
    private static final int CANCEL_MARGIN = Config.getInt("cancelMargin", 4);

    /**
     * How strongly to bias generation toward the player's heading
     * direction.  0.0 = pure Manhattan (360° fill), 1.0 = aggressive
     * forward cone.  See {@link ChunkScheduler} for details.
     */
    private static final float CONE_STRENGTH =
            (float) Config.getDouble("coneStrength", 0.5);

    /**
     * Back-pressure cap: stop enqueuing when the queue exceeds this
     * many tracked tasks.  Prevents runaway memory when the player
     * moves faster than inference can process.
     */
    private static final int MAX_QUEUE_SIZE = Config.getInt("maxQueueSize", 5000);

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private final AtomicBoolean positionReady = new AtomicBoolean(false);
    private volatile Thread workerThread;

    /** Updated each tick from the client thread. */
    private volatile int playerSectionX;
    private volatile int playerSectionZ;

    /** Tracks which (section) positions we've already generated. Thread-safe. */
    private final Set<Long> generatedSections = ConcurrentHashMap.newKeySet();

    /** Cached column conditioning data — avoids redundant noise sampling. Thread-safe. */
    private final ConcurrentHashMap<Long, ColumnContext> columnContextCache =
            new ConcurrentHashMap<>();

    /** Active pipeline queue (set during pipeline execution). */
    private volatile LodGenerationQueue activeQueue;

    /** Stats: how many sections used real vs synthetic conditioning data. Thread-safe. */
    private final AtomicInteger realDataSections = new AtomicInteger();
    private final AtomicInteger syntheticDataSections = new AtomicInteger();
    private final AtomicInteger noiseAccessSections = new AtomicInteger();
    private final AtomicInteger skippedAirSections = new AtomicInteger();

    /** Server-side noise access — null if unavailable (dedicated server). */
    private volatile WorldNoiseAccess noiseAccess;

    /** Server reference for noise access (integrated server in singleplayer). */
    private volatile MinecraftServer server;

    // ------------------------------------------------------------------ //
    //  Lifecycle
    // ------------------------------------------------------------------ //

    /**
     * Start the LOD generation service for a given world.
     *
     * @param world  the Minecraft world (client-side)
     * @param server the Minecraft server (integrated server for singleplayer;
     *               null for dedicated-server clients)
     */
    public void start(World world, MinecraftServer server) {
        if (running.getAndSet(true)) {
            HelloTerrainMod.LOGGER.warn("[LodGen] Service already running");
            return;
        }

        stopRequested.set(false);
        positionReady.set(false);
        generatedSections.clear();
        columnContextCache.clear();
        activeQueue = null;
        realDataSections.set(0);
        syntheticDataSections.set(0);
        noiseAccessSections.set(0);
        skippedAirSections.set(0);
        diagnosticCount.set(0);
        noiseAccess = null;
        this.server = server;

        workerThread = new Thread(() -> runWorker(world), "LODiffusion-Gen");
        workerThread.setDaemon(true);
        workerThread.start();

        HelloTerrainMod.LOGGER.info("[LodGen] Service started");
    }

    /**
     * Stop the service and wait for the worker to finish.
     */
    public void stop() {
        if (!running.get()) return;

        stopRequested.set(true);

        // Signal population done so stage workers can drain and exit
        LodGenerationQueue q = activeQueue;
        if (q != null) q.signalPopulationDone();

        Thread t = workerThread;
        if (t != null) {
            t.interrupt();
            try {
                t.join(5000);
            } catch (InterruptedException ignored) {}
        }
        running.set(false);
        generatedSections.clear();
        columnContextCache.clear();
        if (q != null) q.clear();
        activeQueue = null;
        HelloTerrainMod.LOGGER.info("[LodGen] Service stopped");
    }

    /**
     * Update the player's current section position (called each client tick).
     */
    public void updatePlayerPosition(BlockPos pos) {
        this.playerSectionX = pos.getX() >> 4;
        this.playerSectionZ = pos.getZ() >> 4;
        if (!positionReady.get()) {
            positionReady.set(true);
            HelloTerrainMod.LOGGER.info("[LodGen] Player position initialized: section ({}, {})",
                    playerSectionX, playerSectionZ);
        }
    }

    public boolean isRunning() {
        return running.get();
    }

    // ------------------------------------------------------------------ //
    //  Worker loop
    // ------------------------------------------------------------------ //

    private void runWorker(World world) {
        try {
            HelloTerrainMod.LOGGER.info("[LodGen] Worker starting — waiting for Voxy WorldEngine...");

            // Get Voxy world engine (may need to wait for Voxy to initialize)
            Object worldEngine = waitForWorldEngine(world);
            if (worldEngine == null) {
                HelloTerrainMod.LOGGER.error("[LodGen] Could not obtain Voxy WorldEngine — aborting");
                return;
            }

            HelloTerrainMod.LOGGER.info("[LodGen] Got Voxy WorldEngine — loading model...");

            // Try to bind to the integrated server's noise pipeline for real
            // heightmap / biome / router data at any coordinate.
            noiseAccess = WorldNoiseAccess.tryCreate(server, world);
            if (noiseAccess != null) {
                HelloTerrainMod.LOGGER.info("[LodGen] Using REAL noise access — " +
                        "no synthetic fallback needed");
            } else {
                HelloTerrainMod.LOGGER.warn("[LodGen] Noise access unavailable — " +
                        "will fall back to synthetic heightmap + biome for distant sections");
            }

            // Load model
            ProgressiveModelRunner model = loadModel();
            if (model == null) {
                // ── Heightmap fallback path ──────────────────────────────
                HelloTerrainMod.LOGGER.info(
                        "[LodGen] No ONNX models found — using heightmap fallback generator");

                Object voxyMapper = VoxyCompat.getMapper(worldEngine);
                HeightmapFallbackGenerator.FallbackBlockIds fallbackBlocks =
                        HeightmapFallbackGenerator.resolveBlockIds(voxyMapper);
                int[] fallbackBiomeMappings =
                        HeightmapFallbackGenerator.resolveBiomeMappings(voxyMapper);

                waitForPlayerPosition();
                if (stopRequested.get()) return;

                HelloTerrainMod.LOGGER.info(
                        "[LodGen] Starting FALLBACK generation from player section ({}, {})",
                        playerSectionX, playerSectionZ);

                runFallbackPipeline(world, worldEngine, voxyMapper,
                        fallbackBlocks, fallbackBiomeMappings);
                return;
            }

            // ── Normal ONNX pipeline path ────────────────────────────────
            // Build Voxy block mapper
            Object voxyMapper = VoxyCompat.getMapper(worldEngine);
            VoxyBlockMapper blockMapper = VoxyBlockMapper.build(model.vocabulary(), voxyMapper);
            VoxySectionWriter writer = new VoxySectionWriter(worldEngine, blockMapper);

            HelloTerrainMod.LOGGER.info("[LodGen] Ready — waiting for player position " +
                    "(vocab={}, biomeVoxyId={})", model.vocabulary().size(), blockMapper.defaultBiomeVoxyId());

            // Wait for the client tick to supply the real player position
            waitForPlayerPosition();
            if (stopRequested.get()) return;

            HelloTerrainMod.LOGGER.info("[LodGen] Starting generation from player section ({}, {})",
                    playerSectionX, playerSectionZ);

            // Run the per-stage pipeline
            runPipeline(world, model, writer, blockMapper);

        } catch (Exception e) {
            if (!stopRequested.get()) {
                HelloTerrainMod.LOGGER.error("[LodGen] Worker crashed: {}", e.getMessage(), e);
            }
        } finally {
            running.set(false);
            HelloTerrainMod.LOGGER.info("[LodGen] Worker exited");
        }
    }

    // ------------------------------------------------------------------ //
    //  Per-stage pipeline
    // ------------------------------------------------------------------ //

    /**
     * Run the continuous pipeline: start stage workers, then enter the
     * {@link ChunkScheduler} loop which feeds new sections as the player
     * moves.  The scheduler runs until {@code stopRequested} is set.
     */
    private void runPipeline(World world,
                              ProgressiveModelRunner model,
                              VoxySectionWriter writer,
                              VoxyBlockMapper blockMapper) {

        LodGenerationQueue queue = new LodGenerationQueue();
        this.activeQueue = queue;

        HelloTerrainMod.LOGGER.info(
                "[LodGen] Pipeline starting — {} stage-0 workers + 3 stage workers, radius={}",
                STAGE_0_PARALLELISM, GENERATION_RADIUS);

        // ── Create and start stage worker threads ───────────────────────
        int numWorkers = STAGE_0_PARALLELISM + 3;
        Thread[] workers = new Thread[numWorkers];
        AtomicInteger stage0Active = new AtomicInteger(STAGE_0_PARALLELISM);

        // Stage 0: pool of workers (no parent dependency → safe to parallelize)
        for (int i = 0; i < STAGE_0_PARALLELISM; i++) {
            final int idx = i;
            workers[i] = new Thread(() -> {
                try {
                    runStageWorker(0, queue, model, writer, blockMapper);
                } finally {
                    if (stage0Active.decrementAndGet() == 0) {
                        queue.signalStageComplete(0);
                    }
                }
            }, "LODiffusion-Stage0-" + idx);
            workers[i].setDaemon(true);
        }

        // Stages 1, 2, 3: single-threaded each
        for (int s = 1; s <= 3; s++) {
            final int stage = s;
            int workerIdx = STAGE_0_PARALLELISM + s - 1;
            workers[workerIdx] = new Thread(() -> {
                try {
                    runStageWorker(stage, queue, model, writer, blockMapper);
                } finally {
                    queue.signalStageComplete(stage);
                }
            }, "LODiffusion-Stage" + stage);
            workers[workerIdx].setDaemon(true);
        }

        for (Thread w : workers) w.start();

        // ── Run the continuous chunk scheduler ──────────────────────────
        // The scheduler feeds sections into the queue as the player moves.
        // It blocks until stopRequested is set.
        ChunkScheduler scheduler = new ChunkScheduler(
                queue, stopRequested, generatedSections,
                (w, sx, sz) -> getOrBuildColumnContext(w, sx, sz),
                GENERATION_RADIUS, CANCEL_MARGIN, CONE_STRENGTH, MAX_QUEUE_SIZE);

        scheduler.run(world,
                () -> playerSectionX,
                () -> playerSectionZ,
                (sx, sz) -> tryGetLoadedChunk(world, sx, sz) != null);

        // ── Scheduler exited (stopRequested) — signal pipeline shutdown ─
        queue.signalPopulationDone();

        // Wait for all stage workers to drain and exit
        for (Thread w : workers) {
            try {
                w.join(10_000);
            } catch (InterruptedException e) {
                if (!stopRequested.get()) {
                    HelloTerrainMod.LOGGER.warn("[LodGen] Pipeline interrupted");
                }
                break;
            }
        }

        // Evict column context cache entries beyond 2× radius
        int evicted = ChunkScheduler.evictDistantSections(
                generatedSections, playerSectionX, playerSectionZ,
                GENERATION_RADIUS * 10);
        columnContextCache.clear();

        HelloTerrainMod.LOGGER.info(
                "[LodGen] Pipeline complete — {} done, {} failed " +
                "(noise={}, real={}, synthetic={}, skippedAir={}, evicted={})",
                queue.completedCount(), queue.failedCount(),
                noiseAccessSections.get(), realDataSections.get(),
                syntheticDataSections.get(), skippedAirSections.get(), evicted);
    }

    /**
     * Populate the stage-0 queue with all sections in the generation radius.
     * Sections are prioritized by Manhattan distance (closest first).
     */
    private void populateQueue(LodGenerationQueue queue, World world,
                                int centerX, int centerZ) {
        List<int[]> columns = buildSpiralSections(centerX, centerZ,
                GENERATION_RADIUS, false);

        int enqueued = 0;
        for (int[] col : columns) {
            if (stopRequested.get()) break;

            int sx = col[0], sz = col[1];

            // Skip columns where vanilla has loaded real chunks
            if (tryGetLoadedChunk(world, sx, sz) != null) continue;

            // Get or build column context (cached across Y sections)
            ColumnContext ctx = getOrBuildColumnContext(world, sx, sz);

            // Compute Y range from surface heightmap
            float minH = Float.MAX_VALUE, maxH = -Float.MAX_VALUE;
            for (int lx = 0; lx < 16; lx++) {
                for (int lz = 0; lz < 16; lz++) {
                    float h = ctx.rawHm()[lx][lz];
                    if (h < minH) minH = h;
                    if (h > maxH) maxH = h;
                }
            }

            int minSectionY = Math.max(
                    Math.floorDiv((int) Math.floor(minH), 16) - SURFACE_MARGIN,
                    Y_BASE_SECTION);
            int maxSectionY = Math.min(
                    Math.floorDiv((int) Math.ceil(maxH), 16) + SURFACE_MARGIN,
                    Y_BASE_SECTION + Y_SECTIONS - 1);

            int priority = Math.abs(sx - centerX) + Math.abs(sz - centerZ);

            for (int sy = minSectionY; sy <= maxSectionY; sy++) {
                long key = sectionKey(sx, sy, sz);
                if (generatedSections.contains(key)) continue;

                SectionTask task = new SectionTask(sx, sy, sz, priority, key);
                task.columnContext = ctx;
                if (queue.enqueue(task)) {
                    enqueued++;
                }
            }

            // Track sections we skipped (above/below surface)
            int generatedRange = maxSectionY - minSectionY + 1;
            skippedAirSections.addAndGet(Y_SECTIONS - generatedRange);
        }

        queue.setTotalEnqueued(enqueued);
        queue.signalPopulationDone();

        HelloTerrainMod.LOGGER.info(
                "[LodGen] Enqueued {} sections from {} columns (radius={})",
                enqueued, columns.size(), GENERATION_RADIUS);
    }

    // ------------------------------------------------------------------ //
    //  Stage workers
    // ------------------------------------------------------------------ //

    /**
     * Worker loop for a single pipeline stage.  Drains batches of tasks from
     * the stage's priority queue, processes them in a single batched ONNX call,
     * and promotes them to the next stage.  Falls back to single-sample when
     * only one task is available.  Exits when upstream is done and the queue
     * is permanently empty.
     */
    private void runStageWorker(int stage, LodGenerationQueue queue,
                                 ProgressiveModelRunner model,
                                 VoxySectionWriter writer,
                                 VoxyBlockMapper blockMapper) {
        String threadName = Thread.currentThread().getName();
        HelloTerrainMod.LOGGER.info("[LodGen] {} starting (maxBatch={})",
                threadName, MAX_BATCH_SIZE);

        int processed = 0;

        while (!stopRequested.get()) {
            List<SectionTask> batch;
            try {
                batch = queue.drainStage(stage, MAX_BATCH_SIZE,
                        200, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                break;
            }

            if (batch.isEmpty()) {
                // Check if upstream is done and our queue is permanently empty
                if (queue.isUpstreamDone(stage)) {
                    // One final non-blocking drain to avoid TOCTOU race
                    batch = new ArrayList<>();
                    SectionTask last;
                    while ((last = queue.poll(stage)) != null) {
                        batch.add(last);
                    }
                    if (batch.isEmpty()) break;
                } else {
                    continue;
                }
            }

            // Claim all tasks; filter out any that can't be claimed
            List<SectionTask> claimed = new ArrayList<>(batch.size());
            for (SectionTask task : batch) {
                if (task.claimForProcessing()) {
                    claimed.add(task);
                }
            }
            if (claimed.isEmpty()) continue;

            try {
                if (claimed.size() == 1) {
                    // Single-sample fast path (uses thread-local buffers)
                    processStageTask(stage, claimed.get(0), model, writer, blockMapper);
                } else {
                    // Batched ONNX inference
                    processStageBatch(stage, claimed, model, writer, blockMapper);
                }
                processed += claimed.size();

                if (processed % 500 < claimed.size()) {
                    HelloTerrainMod.LOGGER.info(
                            "[LodGen] {} progress: {} processed (last batch={}), queues: [{}|{}|{}|{}], done: {}",
                            threadName, processed, claimed.size(),
                            queue.stageQueueSize(0), queue.stageQueueSize(1),
                            queue.stageQueueSize(2), queue.stageQueueSize(3),
                            queue.completedCount());
                }
            } catch (Exception e) {
                for (SectionTask task : claimed) {
                    task.markFailed(e.getMessage());
                    queue.markFailed();
                }
                if (!stopRequested.get()) {
                    HelloTerrainMod.LOGGER.warn("[LodGen] {} batch failed ({} tasks): {}",
                            threadName, claimed.size(), e.getMessage(), e);
                }
            }
        }

        HelloTerrainMod.LOGGER.info("[LodGen] {} exiting — processed {} tasks",
                threadName, processed);
    }

    /**
     * Process a single section task at a given pipeline stage.
     *
     * <p><b>Progressive LOD write path:</b> Every stage writes its block
     * predictions directly to the corresponding Voxy storage level via
     * {@link VoxySectionWriter#writeLodSection}.  Stages 0-2 also propagate
     * binary solid parent data to the next stage.
     *
     * <ul>
     *   <li><b>Stage 0</b> (Init→LOD4, 1³) → write to Voxy lvl 4, promote</li>
     *   <li><b>Stage 1</b> (LOD4→LOD3, 2³) → write to Voxy lvl 3, promote</li>
     *   <li><b>Stage 2</b> (LOD3→LOD2, 4³) → write to Voxy lvl 2, promote</li>
     *   <li><b>Stage 3</b> (LOD2→LOD1, 8³) → write to Voxy lvl 1, also
     *       writes 2×-upsampled 16³ via legacy path for completeness</li>
     * </ul>
     */
    private void processStageTask(int stage, SectionTask task,
                                   ProgressiveModelRunner model,
                                   VoxySectionWriter writer,
                                   VoxyBlockMapper blockMapper)
            throws Exception {

        ColumnContext ctx = task.columnContext;
        int yIndex = task.sectionY - Y_BASE_SECTION;

        ProgressiveModelRunner.StageOutput output = model.generateStage(
                stage, ctx.hp5(), ctx.biomeIdx(), yIndex, task.parentFlat);

        // Map stage → Voxy level: stage 0 → lvl4, stage 1 → lvl3, etc.
        int voxyLvl = COARSEST_LOD - stage;  // 4, 3, 2, 1

        // Translate canonical biome IDs → Voxy biome IDs
        int[][] biomeVoxyIds = buildBiomeVoxyIds(ctx.biomeIdx(), blockMapper);

        // Write this stage's output to Voxy via insertUpdate (upsampled to 16³).
        // This properly propagates nonEmptyChildren through the octree, making
        // each LOD level visible immediately as it completes.
        float[][][][][] nativeLogits = output.blockLogits();
        if (nativeLogits != null) {
            // Apply heightmap clip at the native resolution
            if (HEIGHTMAP_CLIP && ctx.rawHm() != null) {
                applyHeightmapClipScaled(nativeLogits, ctx.rawHm(),
                        task.sectionY, voxyLvl);
            }

            writer.writeUpsampledSection(nativeLogits,
                    model.config().effectiveBlockVocabSize(),
                    voxyLvl, task.sectionX, task.sectionY, task.sectionZ,
                    biomeVoxyIds);
        }

        if (stage < 3) {
            // Intermediate stage: promote to next stage with solid parent
            task.promoteToNextStage(output.solidParentFlat());
            activeQueue.promoteToNextStage(task);
        } else {
            // Final stage — diagnostics only (insertUpdate above handles the write)
            InferenceResult result = output.finalResult();

            // Diagnostics for first few sections
            if (diagnosticCount.get() < 10) {
                logDiagnostics(result, model.config(), model.vocabulary(), blockMapper,
                        task.sectionX, task.sectionY, task.sectionZ,
                        output.elapsedMs(), 1 /* finest LOD */,
                        yIndex, 0);
                diagnosticCount.incrementAndGet();
            }

            task.markReady();
            generatedSections.add(task.key);
            activeQueue.markCompleted();
        }
    }

    /**
     * Process a batch of section tasks at a given pipeline stage using a
     * single batched ONNX inference call.
     *
     * <p>This is the batched counterpart to {@link #processStageTask}: it
     * collects conditioning data from all tasks, calls
     * {@link ProgressiveModelRunner#generateStageBatch}, then distributes
     * the per-section results identically to the single-sample path
     * (write to Voxy, promote to next stage, etc.).
     */
    private void processStageBatch(int stage, List<SectionTask> tasks,
                                    ProgressiveModelRunner model,
                                    VoxySectionWriter writer,
                                    VoxyBlockMapper blockMapper) throws Exception {

        int batchSize = tasks.size();

        // ── Collect conditioning arrays from all tasks ──────────────────
        float[][][] hp5Rows   = new float[batchSize][][];
        int[][][]   biomeIdxs = new int[batchSize][][];
        int[]       yIndices  = new int[batchSize];
        float[][]   parentFlats = new float[batchSize][];

        for (int i = 0; i < batchSize; i++) {
            SectionTask task = tasks.get(i);
            ColumnContext ctx = task.columnContext;
            hp5Rows[i]    = ctx.hp5();
            biomeIdxs[i]  = ctx.biomeIdx();
            yIndices[i]   = task.sectionY - Y_BASE_SECTION;
            parentFlats[i] = task.parentFlat;
        }

        // ── Single batched ONNX call ────────────────────────────────────
        List<ProgressiveModelRunner.StageOutput> outputs =
                model.generateStageBatch(stage, hp5Rows, biomeIdxs,
                        yIndices, parentFlats);

        // ── Distribute per-section results ──────────────────────────────
        int voxyLvl = COARSEST_LOD - stage;

        for (int i = 0; i < batchSize; i++) {
            SectionTask task = tasks.get(i);
            ProgressiveModelRunner.StageOutput output = outputs.get(i);
            ColumnContext ctx = task.columnContext;

            int[][] biomeVoxyIds = buildBiomeVoxyIds(ctx.biomeIdx(), blockMapper);

            float[][][][][] nativeLogits = output.blockLogits();
            if (nativeLogits != null) {
                if (HEIGHTMAP_CLIP && ctx.rawHm() != null) {
                    applyHeightmapClipScaled(nativeLogits, ctx.rawHm(),
                            task.sectionY, voxyLvl);
                }
                writer.writeUpsampledSection(nativeLogits,
                        model.config().effectiveBlockVocabSize(),
                        voxyLvl, task.sectionX, task.sectionY, task.sectionZ,
                        biomeVoxyIds);
            }

            if (stage < 3) {
                task.promoteToNextStage(output.solidParentFlat());
                activeQueue.promoteToNextStage(task);
            } else {
                InferenceResult result = output.finalResult();

                if (diagnosticCount.get() < 10) {
                    int yIndex = task.sectionY - Y_BASE_SECTION;
                    logDiagnostics(result, model.config(), model.vocabulary(),
                            blockMapper, task.sectionX, task.sectionY, task.sectionZ,
                            output.elapsedMs(), 1, yIndex, 0);
                    diagnosticCount.incrementAndGet();
                }

                task.markReady();
                generatedSections.add(task.key);
                activeQueue.markCompleted();
            }
        }
    }

    /**
     * Force air above the surface heightmap by boosting the air logit
     * (channel 0) to a large value, ensuring argmax selects air.
     */
    private static void applyHeightmapClip(float[][][][][] logits, float[][] rawHm,
                                            int sectionY) {
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                float surfaceY = rawHm[lx][lz];
                for (int ly = 0; ly < 16; ly++) {
                    int worldY = sectionY * 16 + ly;
                    if (worldY >= surfaceY) {
                        // Set air logit (channel 0) to a large value so it
                        // wins the argmax over any solid-block logit.
                        logits[0][0][ly][lz][lx] = 100f;
                    }
                }
            }
        }
    }

    /**
     * Heightmap clip for native-resolution logits at a specific LOD level.
     *
     * <p>At LOD level {@code voxyLvl}, each voxel covers {@code 2^voxyLvl}
     * blocks along each axis.  A voxel is clipped to air if its <em>lowest</em>
     * block Y is at or above the surface height for the center of its XZ
     * footprint.
     *
     * @param logits    [1][N][D][D][D] where D = 16 >> voxyLvl
     * @param rawHm     [16][16] surface heightmap in [x][z] order
     * @param sectionY  L0 section Y coordinate
     * @param voxyLvl   Voxy storage level (1-4)
     */
    private static void applyHeightmapClipScaled(float[][][][][] logits, float[][] rawHm,
                                                   int sectionY, int voxyLvl) {
        int cellsPerAxis = 16 >> voxyLvl;  // 8,4,2,1
        int blocksPerVoxel = 1 << voxyLvl; // 2,4,8,16

        for (int lx = 0; lx < cellsPerAxis; lx++) {
            for (int lz = 0; lz < cellsPerAxis; lz++) {
                // Sample heightmap at the center of this voxel's XZ footprint
                int hmX = Math.min(lx * blocksPerVoxel + blocksPerVoxel / 2, 15);
                int hmZ = Math.min(lz * blocksPerVoxel + blocksPerVoxel / 2, 15);
                float surfaceY = rawHm[hmX][hmZ];

                for (int ly = 0; ly < cellsPerAxis; ly++) {
                    // Lowest block Y covered by this voxel
                    int worldY = sectionY * 16 + ly * blocksPerVoxel;
                    if (worldY >= surfaceY) {
                        logits[0][0][ly][lz][lx] = 100f;
                    }
                }
            }
        }
    }

    /**
     * Translate canonical biome IDs to Voxy biome IDs for a column.
     */
    private static int[][] buildBiomeVoxyIds(int[][] biomeIdx,
                                              VoxyBlockMapper blockMapper) {
        int[][] biomeVoxyIds = new int[16][16];
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                biomeVoxyIds[lx][lz] = blockMapper.getVoxyBiomeId(biomeIdx[lx][lz]);
            }
        }
        return biomeVoxyIds;
    }

    /** Counter for detailed diagnostics (reset per LOD pass). Thread-safe. */
    private final AtomicInteger diagnosticCount = new AtomicInteger();

    // ------------------------------------------------------------------ //
    //  Per-column conditioning context
    // ------------------------------------------------------------------ //

    /**
     * Pre-sampled conditioning data for a single 16×16 column (chunk).
     * Sampled once per column and reused for all Y sections in that column.
     */
    record ColumnContext(
        float[][] rawHm,          // [16][16] surface heightmap in block Y
        int[][]   biomeIdx,       // [16][16] biome indices
        float[][] hp5,            // [5][256] height-planes (row-major)
        float[][] r6,             // [6][256] router6 values (row-major)
        float[][] oceanFloorHm    // [16][16] ocean/river floor block-Y (may be null)
    ) {}

    /**
     * Build the conditioning context for a column, using the best available
     * data source.
     *
     * <p>Priority:
     * <ol>
     *   <li>{@link WorldNoiseAccess} — real heightmap + router6 + biome
     *       at any coordinate (no chunk needed)</li>
     *   <li>Loaded chunk — real heightmap + biome, approximate router6</li>
     *   <li>Synthetic — sine-wave heightmap + constant biome (last resort)</li>
     * </ol>
     */
    private ColumnContext buildColumnContext(World world, int sectionX, int sectionZ) {
        float[][] rawHm;
        int[][]   biomeIdx;
        float[][] hp5;
        float[][] r6;
        float[][] oceanFloorHm = null;

        if (noiseAccess != null) {
            // *** PRIMARY PATH: Real noise data at any coordinate ***
            // sampleFromNoise() now returns rawHm inside AnchorInputs — no second
            // sampleHeightmap() call needed (eliminates 256 duplicate getHeight() calls).
            AnchorSampler.AnchorInputs anchor =
                    AnchorSampler.sampleFromNoise(noiseAccess, sectionX, sectionZ);
            rawHm        = anchor.rawHm();
            biomeIdx     = anchor.biomeIdx();
            hp5          = anchor.heightPlanes5();
            r6           = anchor.router6();
            oceanFloorHm = anchor.oceanFloorHm();
            noiseAccessSections.incrementAndGet();
            if (diagnosticCount.get() < 3) {
                HelloTerrainMod.LOGGER.info(
                        "[LodGen] Using NOISE ACCESS data for column ({},{}) — " +
                        "real heightmap + router6 + biome",
                        sectionX, sectionZ);
            }
        } else {
            Chunk chunk = tryGetLoadedChunk(world, sectionX, sectionZ);
            if (chunk != null) {
                rawHm    = AnchorSampler.sampleHeightmap(chunk);
                biomeIdx = AnchorSampler.sampleBiomes(chunk);
                realDataSections.incrementAndGet();
                if (diagnosticCount.get() < 3) {
                    HelloTerrainMod.LOGGER.info(
                            "[LodGen] Using REAL chunk data for column ({},{})"
                            + " — WARNING: router6 is APPROXIMATE (no noise access)",
                            sectionX, sectionZ);
                }
            } else {
                // Last resort — synthetic (should rarely happen with noise access)
                rawHm    = buildHeightmap(sectionX, sectionZ);
                biomeIdx = new int[16][16];
                for (int[] row : biomeIdx) java.util.Arrays.fill(row, 1);
                syntheticDataSections.incrementAndGet();
                if (diagnosticCount.get() < 3) {
                    HelloTerrainMod.LOGGER.info(
                            "[LodGen] Using SYNTHETIC data for column ({},{}) — " +
                            "chunk not loaded, no noise access.  " +
                            "Router6 is APPROXIMATE — quality WILL be degraded.",
                            sectionX, sectionZ);
                }
            }
            hp5 = AnchorSampler.computeHeightPlanes(rawHm, null);  // no ocean floor in synthetic path
            @SuppressWarnings("deprecation")
            float[][] approxR6 = AnchorSampler.approximateRouter6(biomeIdx, rawHm);
            r6 = approxR6;
        }

        return new ColumnContext(rawHm, biomeIdx, hp5, r6, oceanFloorHm);
    }

    /**
     * Get or build column context with caching. Thread-safe.
     * Avoids redundant noise sampling when multiple Y sections share
     * the same column.
     */
    private ColumnContext getOrBuildColumnContext(World world, int sx, int sz) {
        long key = ((long) (sx & 0xFFFF) << 16) | (sz & 0xFFFFL);
        return columnContextCache.computeIfAbsent(key, k -> buildColumnContext(world, sx, sz));
    }

    // ------------------------------------------------------------------ //
    //  Coarsen & diagnostics
    // ------------------------------------------------------------------ //

    /**
     * Coarsen a 16³ block-logits tensor to an 8³ binary parent occupancy grid.
     *
     * <p>Uses max-pooling over 2×2×2 blocks via argmax: if <em>any</em> voxel
     * in the block predicts solid (argmax != 0), the parent cell is set to 1.0.
     *
     * @param blockLogits model output [1][N][16][16][16]
     * @return [8][8][8] binary parent (0.0 = air, 1.0 = solid)
     */
    static float[][][] coarsenToParent(float[][][][][] blockLogits) {
        int vocabSize = blockLogits[0].length;
        float[][][] parent = new float[8][8][8];
        for (int px = 0; px < 8; px++) {
            for (int py = 0; py < 8; py++) {
                for (int pz = 0; pz < 8; pz++) {
                    boolean anySolid = false;
                    outer:
                    for (int dx = 0; dx < 2; dx++)
                        for (int dy = 0; dy < 2; dy++)
                            for (int dz = 0; dz < 2; dz++) {
                                int vy = px * 2 + dx, vz = py * 2 + dy, vx = pz * 2 + dz;
                                // argmax: is any channel > 0 (air channel)?
                                float airLogit = blockLogits[0][0][vy][vz][vx];
                                for (int c = 1; c < vocabSize; c++) {
                                    if (blockLogits[0][c][vy][vz][vx] > airLogit) {
                                        anySolid = true;
                                        break outer;
                                    }
                                }
                            }
                    parent[px][py][pz] = anySolid ? 1.0f : 0.0f;
                }
            }
        }
        return parent;
    }

    /**
     * Build a heightmap-based initial parent for the coarsest LOD pass.
     *
     * <p>Each cell in the 8³ grid is set to 1.0 (solid) if its center Y
     * coordinate is below the heightmap value at that column, 0.0 (air)
     * otherwise.  This gives the model a reasonable starting structure
     * instead of an empty (all-air) grid.
     */

    /**
     * Log detailed model-output diagnostics for a section.
     */
    private void logDiagnostics(InferenceResult result,
                                 ModelConfig config,
                                 BlockVocabulary vocabulary,
                                 VoxyBlockMapper blockMapper,
                                 int sectionX, int sectionY, int sectionZ,
                                 long elapsedMs, int lod,
                                 int yIndex, int parentSolid) {
        float[][][][][] logits = result.blockLogits();
        int vocabSize = config.effectiveBlockVocabSize();

        // Argmax solid count (air = class 0)
        int solidCount = 0;
        for (int d0 = 0; d0 < 16; d0++)
            for (int d1 = 0; d1 < 16; d1++)
                for (int d2 = 0; d2 < 16; d2++) {
                    int best = 0;
                    float bestVal = logits[0][0][d0][d1][d2];
                    for (int b = 1; b < Math.min(vocabSize, logits[0].length); b++) {
                        float v = logits[0][b][d0][d1][d2];
                        if (v > bestVal) {
                            bestVal = v;
                            best = b;
                        }
                    }
                    if (best > 0) solidCount++;
                }

        // Block logit stats
        float logitMin = Float.MAX_VALUE, logitMax = -Float.MAX_VALUE;
        for (int b = 0; b < Math.min(vocabSize, logits[0].length); b++)
            for (int d0 = 0; d0 < 16; d0++)
                for (int d1 = 0; d1 < 16; d1++)
                    for (int d2 = 0; d2 < 16; d2++) {
                        float v = logits[0][b][d0][d1][d2];
                        if (v < logitMin) logitMin = v;
                        if (v > logitMax) logitMax = v;
                    }

        // Top predicted block at center voxel (8,8,8)
        int centerBest = 0;
        float centerBestVal = logits[0][0][8][8][8];
        for (int b = 1; b < Math.min(vocabSize, logits[0].length); b++) {
            float v = logits[0][b][8][8][8];
            if (v > centerBestVal) {
                centerBestVal = v;
                centerBest = b;
            }
        }

        int voxyId = blockMapper.getVoxyBlockId(centerBest);
        String blockName = centerBest < vocabulary.size()
                ? vocabulary.getName(centerBest) : "???";

        HelloTerrainMod.LOGGER.info(
                "[LodGen] DIAG stage {} ({},{},{}): " +
                "yIdx={} | solid={}/4096 | " +
                "logit range=[{},{}] | " +
                "center: idx={} '{}' voxyId={} | " +
                "{}ms",
                lod, sectionX, sectionY, sectionZ,
                yIndex, solidCount,
                logitMin, logitMax,
                centerBest, blockName, voxyId,
                elapsedMs);
    }

    // ------------------------------------------------------------------ //
    //  Input building (position-dependent synthetic conditioning)
    // ------------------------------------------------------------------ //

    /** Sea level in block Y coordinates. */
    private static final float SEA_LEVEL = 62f;

    /** Amplitude of terrain height variation (blocks). */
    private static final float HEIGHT_AMPLITUDE = 24f;

    /**
     * Build a raw heightmap (in block Y coordinates) for a 16×16 section
     * column at the given section (x, z).  Uses deterministic multi-octave
     * sine/cosine noise so that adjacent sections share consistent terrain
     * shape.
     *
     * @return float[16][16] of raw block-Y heights (approx 40–90 range)
     */
    private float[][] buildHeightmap(int sectionX, int sectionZ) {
        float[][] hm = new float[16][16];
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                // Global block coordinates
                float bx = sectionX * 16f + lx;
                float bz = sectionZ * 16f + lz;

                // Multi-octave sine noise for gentle rolling hills
                float h = SEA_LEVEL;
                h += HEIGHT_AMPLITUDE * 0.50f * (float) Math.sin(bx * 0.005 + 1.7)
                                              * (float) Math.cos(bz * 0.007 + 0.3);
                h += HEIGHT_AMPLITUDE * 0.25f * (float) Math.sin(bx * 0.013 + 3.1)
                                              * (float) Math.sin(bz * 0.011 + 2.2);
                h += HEIGHT_AMPLITUDE * 0.12f * (float) Math.cos(bx * 0.037 + 0.9)
                                              * (float) Math.sin(bz * 0.029 + 4.1);
                // Clamp to valid MC range
                hm[lx][lz] = Math.max(0f, Math.min(320f, h));
            }
        }
        return hm;
    }

    // ------------------------------------------------------------------ //
    //  Section spiral ordering
    // ------------------------------------------------------------------ //

    /**
     * Build a list of (x, z) section coordinates ordered by distance
     * from the center, limited to the given radius.
     *
     * <p>Every pass covers a <em>full disc</em> — finer LOD passes
     * naturally overwrite our earlier coarser data.  Voxy-native sections
     * (from real chunk loading) are protected in {@link VoxySectionWriter}.
     *
     * @param distantFirst if true, sort furthest-from-center first so
     *                     distant horizon terrain appears immediately.
     */
    private List<int[]> buildSpiralSections(int centerX, int centerZ,
                                             int radius, boolean distantFirst) {
        List<int[]> sections = new ArrayList<>();
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dz = -radius; dz <= radius; dz++) {
                sections.add(new int[]{centerX + dx, centerZ + dz});
            }
        }
        // Sort by Manhattan distance — ascending (center-first) or
        // descending (horizon-first) depending on the pass.
        Comparator<int[]> cmp = Comparator.comparingInt(s ->
                Math.abs(s[0] - centerX) + Math.abs(s[1] - centerZ));
        sections.sort(distantFirst ? cmp.reversed() : cmp);
        return sections;
    }

    // ------------------------------------------------------------------ //
    //  Helpers
    // ------------------------------------------------------------------ //

    /**
     * Try to get a loaded chunk from the world without blocking or
     * triggering generation.  Returns null if the chunk is not currently
     * loaded in the client (or server) chunk manager.
     *
     * <p>This is called from the worker thread; access is read-only so
     * it is safe on both client and server chunk managers.
     */
    private Chunk tryGetLoadedChunk(World world, int chunkX, int chunkZ) {
        try {
            return world.getChunkManager().getChunk(
                    chunkX, chunkZ, ChunkStatus.FULL, false);
        } catch (Exception e) {
            // Chunk manager threw — treat as not loaded
            return null;
        }
    }

    /**
     * Pack section coordinates into a single long key for deduplication.
     * Each axis uses 20 bits, supporting ±524,287 sections.
     */
    static long sectionKey(int x, int y, int z) {
        return ((long) (x & 0xFFFFF) << 40) | ((long) (y & 0xFFFFF) << 20) | (z & 0xFFFFFL);
    }

    /**
     * Block until the client tick handler has supplied the player's real position.
     */
    private void waitForPlayerPosition() {
        for (int i = 0; i < 300; i++) {   // 30 seconds max
            if (stopRequested.get()) return;
            if (positionReady.get()) return;
            try {
                Thread.sleep(100);
            } catch (InterruptedException e) {
                return;
            }
        }
        HelloTerrainMod.LOGGER.warn("[LodGen] Timed out waiting for player position — using (0, 0)");
    }

    /**
     * Wait for the Voxy WorldEngine to become available (with timeout).
     */
    private Object waitForWorldEngine(World world) {
        for (int attempt = 0; attempt < 60; attempt++) {
            if (stopRequested.get()) return null;

            Object engine = VoxyCompat.getWorldEngine(world);
            if (engine != null) return engine;

            HelloTerrainMod.LOGGER.debug("[LodGen] Waiting for Voxy WorldEngine (attempt {})", attempt);
            try {
                Thread.sleep(1000);
            } catch (InterruptedException e) {
                return null;
            }
        }
        return null;
    }

    // ------------------------------------------------------------------ //
    //  Heightmap fallback pipeline
    // ------------------------------------------------------------------ //

    /**
     * Ultra-fast fallback pipeline that generates terrain from heightmap data
     * alone, without any ONNX model.  Processes one column at a time on a
     * single thread — I/O ({@code insertUpdate}) is the bottleneck, not compute.
     *
     * <p>Reuses all existing infrastructure: spiral ordering, column context
     * caching, surface Y-range filtering, and deduplication.
     */
    private void runFallbackPipeline(World world, Object worldEngine,
                                      Object voxyMapper,
                                      HeightmapFallbackGenerator.FallbackBlockIds blockIds,
                                      int[] biomeVoxyMappings) {
        int totalSections = 0;
        int skippedAir = 0;
        int skippedExisting = 0;
        int columnsProcessed = 0;
        long startTime = System.currentTimeMillis();

        HelloTerrainMod.LOGGER.info(
                "[LodGen] Fallback pipeline starting — continuous mode, radius={}",
                GENERATION_RADIUS);

        // ── Continuous loop: re-spiral from player position each pass ───
        while (!stopRequested.get()) {
            int centerX = playerSectionX;
            int centerZ = playerSectionZ;

            List<int[]> columns = buildSpiralSections(centerX, centerZ,
                    GENERATION_RADIUS, false);

            boolean anyNew = false;

            for (int[] col : columns) {
                if (stopRequested.get()) break;

                int sx = col[0], sz = col[1];

                // If player moved far, restart spiral from new position
                int drift = Math.abs(playerSectionX - centerX)
                          + Math.abs(playerSectionZ - centerZ);
                if (drift > 2) break;

                // NOTE: We intentionally do NOT skip columns loaded by vanilla.
                // The per-section sectionExists() check below prevents overwriting
                // any sections Voxy has already ingested from real chunks, and
                // filling the rest avoids a visible gap between vanilla render
                // distance and the LOD terrain.

                // Get or build column context (cached across Y sections)
                ColumnContext ctx = getOrBuildColumnContext(world, sx, sz);

                // Build per-column Voxy biome IDs
                int[][] biomeVoxyIds = new int[16][16];
                for (int lx = 0; lx < 16; lx++) {
                    for (int lz = 0; lz < 16; lz++) {
                        int canonical = ctx.biomeIdx()[lx][lz];
                        biomeVoxyIds[lx][lz] = (canonical >= 0 && canonical < biomeVoxyMappings.length)
                                ? biomeVoxyMappings[canonical] : 0;
                    }
                }

                // Compute Y range from surface heightmap
                float minH = Float.MAX_VALUE, maxH = -Float.MAX_VALUE;
                for (int lx = 0; lx < 16; lx++) {
                    for (int lz = 0; lz < 16; lz++) {
                        float h = ctx.rawHm()[lx][lz];
                        if (h < minH) minH = h;
                        if (h > maxH) maxH = h;
                    }
                }

                // Extend range down to cover water if surface is below sea level
                float effectiveMax = Math.max(maxH, HeightmapFallbackGenerator.SEA_LEVEL);

                int minSectionY = Math.max(
                        Math.floorDiv((int) Math.floor(minH), 16) - SURFACE_MARGIN,
                        Y_BASE_SECTION);
                int maxSectionY = Math.min(
                        Math.floorDiv((int) Math.ceil(effectiveMax), 16) + SURFACE_MARGIN,
                        Y_BASE_SECTION + Y_SECTIONS - 1);

                // Process all Y sections in this column
                for (int sy = minSectionY; sy <= maxSectionY; sy++) {
                    if (stopRequested.get()) break;

                    long key = sectionKey(sx, sy, sz);
                    if (generatedSections.contains(key)) continue;

                    // Skip if Voxy already has real data for this section
                    if (VoxyCompat.sectionExists(worldEngine, sx, sy, sz)) {
                        skippedExisting++;
                        generatedSections.add(key);
                        continue;
                    }

                    Object section = HeightmapFallbackGenerator.generateSection(
                            sx, sy, sz, ctx.rawHm(), ctx.oceanFloorHm(),
                            ctx.biomeIdx(), biomeVoxyIds, blockIds, voxyMapper);

                    if (section == null) {
                        skippedAir++;
                    } else {
                        VoxyCompat.insertUpdate(worldEngine, section);
                        totalSections++;
                        anyNew = true;
                    }

                    generatedSections.add(key);
                }

                columnsProcessed++;

                // Progress logging
                if (columnsProcessed % 100 == 0) {
                    long elapsed = System.currentTimeMillis() - startTime;
                    double sectionsPerSec = totalSections > 0
                            ? totalSections / (elapsed / 1000.0) : 0;
                    HelloTerrainMod.LOGGER.info(
                            "[LodGen] Fallback progress: {} columns, {} sections written, "
                            + "{} air-skipped, {} existing-skipped ({} sec, {} sections/s)",
                            columnsProcessed, totalSections,
                            skippedAir, skippedExisting,
                            elapsed / 1000, (int) sectionsPerSec);
                }

                // Track skipped sections above/below surface
                int generatedRange = maxSectionY - minSectionY + 1;
                skippedAirSections.addAndGet(Y_SECTIONS - generatedRange);
            }

            // If nothing new was generated (all in radius already done), idle
            if (!anyNew && !stopRequested.get()) {
                try {
                    Thread.sleep(500);
                } catch (InterruptedException e) {
                    if (!stopRequested.get()) {
                        HelloTerrainMod.LOGGER.warn("[LodGen] Fallback interrupted");
                    }
                    break;
                }
            }

            // Periodically evict distant tracked sections to bound memory
            ChunkScheduler.evictDistantSections(
                    generatedSections, playerSectionX, playerSectionZ,
                    GENERATION_RADIUS * 10);
        }

        // Free cached column context
        columnContextCache.clear();

        long elapsed = System.currentTimeMillis() - startTime;
        HelloTerrainMod.LOGGER.info(
                "[LodGen] Fallback pipeline stopped — {} sections in {}.{}s "
                + "({} columns, {} air-skipped, {} existing-skipped)",
                totalSections, elapsed / 1000, elapsed % 1000,
                columnsProcessed, skippedAir, skippedExisting);
    }

    /**
     * Load the ONNX model (same lazy-load as OnnxTerrainGenerator).
     */
    private ProgressiveModelRunner loadModel() {
        try {
            java.nio.file.Path modelDir = Config.modelDir();
            HelloTerrainMod.LOGGER.info("[LodGen] Loading progressive models from {}...", modelDir);
            return ProgressiveModelRunner.loadAll(modelDir);
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[LodGen] Model load failed: {}", e.getMessage(), e);
            return null;
        }
    }
}
