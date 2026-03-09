package com.rhythmatician.lodiffusion.voxy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
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
 * <h3>Architecture — Progressive LOD Passes</h3>
 * <p>The ONNX model is a <em>refinement machine</em>.  Generation runs in
 * <b>4 progressive passes</b>, one per LOD level, each covering a different
 * radius around the player:
 *
 * <ol>
 *   <li><b>LOD 4</b> (radius 16) — coarsest pass, covers the horizon fast.
 *       Uses a heightmap-based initial parent.  Result is pushed to Voxy
 *       immediately so terrain appears quickly.</li>
 *   <li><b>LOD 3</b> (radius 12) — takes the coarsened LOD 4 output as
 *       parent, refines it.  Overwrites the LOD 4 sections in Voxy.</li>
 *   <li><b>LOD 2</b> (radius 8) — refines LOD 3 output.</li>
 *   <li><b>LOD 1</b> (radius 4) — finest pass, closest to the player.
 *       Final quality terrain.</li>
 * </ol>
 *
 * <p>Each pass coarsens its 16³ air-mask output to 8³ and caches it so
 * the next pass can use it as the parent occupancy input.  Voxy's
 * {@code insertUpdate} handles the internal mip pyramid automatically.
 */
public final class LodGenerationService {

    /** Model LOD token range. */
    private static final int COARSEST_LOD = 4;
    private static final int FINEST_LOD   = 1;

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
     * Number of parallel worker threads for the coarsest LOD pass.
     * Higher values speed up initial terrain generation at the cost of
     * more CPU usage.  Capped at 4 to avoid starving the game thread.
     */
    private static final int LOD4_PARALLELISM =
            Math.min(Runtime.getRuntime().availableProcessors(), 4);

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private final AtomicBoolean positionReady = new AtomicBoolean(false);
    private volatile Thread workerThread;

    /** Updated each tick from the client thread. */
    private volatile int playerSectionX;
    private volatile int playerSectionZ;

    /** Tracks which (section, lod) combos we've already generated. Thread-safe. */
    private final Set<Long> generatedSections = ConcurrentHashMap.newKeySet();

    /** Coarsened parent cache: posKey → float[8][8][8] from previous LOD pass. */
    private final Map<Long, float[][][]> parentCache = new HashMap<>();

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
        parentCache.clear();
        realDataSections.set(0);
        syntheticDataSections.set(0);
        noiseAccessSections.set(0);
        skippedAirSections.set(0);
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
        Thread t = workerThread;
        if (t != null) {
            t.interrupt();
            try {
                t.join(5000);
            } catch (InterruptedException ignored) {}
        }
        running.set(false);
        generatedSections.clear();
        parentCache.clear();
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

            // Load model
            ProgressiveModelRunner model = loadModel();
            if (model == null) {
                HelloTerrainMod.LOGGER.error("[LodGen] ONNX model failed to load — aborting");
                return;
            }

            // Build Voxy block mapper
            Object voxyMapper = VoxyCompat.getMapper(worldEngine);
            VoxyBlockMapper blockMapper = VoxyBlockMapper.build(model.vocabulary(), voxyMapper);
            VoxySectionWriter writer = new VoxySectionWriter(worldEngine, blockMapper);

            HelloTerrainMod.LOGGER.info("[LodGen] Ready — waiting for player position " +
                    "(vocab={}, biomeVoxyId={})", model.vocabulary().size(), blockMapper.defaultBiomeVoxyId());

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

            // Wait for the client tick to supply the real player position
            waitForPlayerPosition();
            if (stopRequested.get()) return;

            HelloTerrainMod.LOGGER.info("[LodGen] Starting generation from player section ({}, {})",
                    playerSectionX, playerSectionZ);

            // Main generation loop: coarsest LODs first, then refine
            generateProgressiveLods(world, model, writer, blockMapper);

        } catch (Exception e) {
            if (!stopRequested.get()) {
                HelloTerrainMod.LOGGER.error("[LodGen] Worker crashed: {}", e.getMessage(), e);
            }
        } finally {
            running.set(false);
            HelloTerrainMod.LOGGER.info("[LodGen] Worker exited");
        }
    }

    /**
     * Generate terrain in progressive LOD passes, coarsest-to-finest.
     *
     * <p>Each pass covers a decreasing radius: LOD 4 sweeps the horizon
     * quickly, while LOD 1 only refines the area closest to the player.
     * Each section's result is pushed to Voxy immediately, so coarse
     * terrain appears fast and gets refined progressively.
     */
    private void generateProgressiveLods(World world,
                                          ProgressiveModelRunner model,
                                          VoxySectionWriter writer,
                                          VoxyBlockMapper blockMapper) {
        int centerX = playerSectionX;
        int centerZ = playerSectionZ;

        HelloTerrainMod.LOGGER.info(
                "[LodGen] Starting progressive generation — " +
                "{} passes (LOD {}→{}) around ({}, {})  contract={}  parallelism={}",
                COARSEST_LOD - FINEST_LOD + 1,
                COARSEST_LOD, FINEST_LOD, centerX, centerZ,
                "v3.progressive", LOD4_PARALLELISM);

        for (int lod = COARSEST_LOD; lod >= FINEST_LOD; lod--) {
            if (stopRequested.get()) return;

            int radius = PASS_RADIUS[lod];
            boolean distantFirst = false;
            List<int[]> columns = buildSpiralSections(
                    centerX, centerZ, radius, distantFirst);
            AtomicInteger passCount = new AtomicInteger();
            diagnosticCount.set(0);

            HelloTerrainMod.LOGGER.info(
                    "[LodGen] LOD {} pass — radius={}, ~{} columns, " +
                    "distant-first={}, × {} Y = ~{} sections",
                    lod, radius, columns.size(),
                    distantFirst, Y_SECTIONS,
                    columns.size() * Y_SECTIONS);

            boolean useParallel = lod == COARSEST_LOD && LOD4_PARALLELISM > 1;

            if (useParallel) {
                // ── Multi-threaded pass for coarsest LOD ─────────────
                // LOD 4 covers the widest radius with the cheapest model.
                // Parallelising column processing fills the horizon faster.
                ExecutorService executor = Executors.newFixedThreadPool(
                        LOD4_PARALLELISM, r -> {
                            Thread t = new Thread(r, "LODiffusion-LOD4");
                            t.setDaemon(true);
                            return t;
                        });

                final int lodFinal = lod;
                List<Future<Integer>> futures = new ArrayList<>();
                for (int[] col : columns) {
                    final int sx = col[0], sz = col[1];
                    futures.add(executor.submit(() ->
                            processColumn(world, model, writer, blockMapper,
                                    sx, sz, lodFinal)));
                }

                // Collect results as they complete (futures are ordered
                // center-first from the spiral, so progress is natural)
                for (Future<Integer> f : futures) {
                    if (stopRequested.get()) break;
                    try {
                        int count = f.get();
                        int total = passCount.addAndGet(count);
                        if (count > 0 && total % 200 < count) {
                            HelloTerrainMod.LOGGER.info(
                                    "[LodGen] LOD {} progress: {} sections " +
                                    "(skipped {} air)",
                                    lodFinal, total, skippedAirSections.get());
                        }
                    } catch (Exception e) {
                        if (!stopRequested.get()) {
                            HelloTerrainMod.LOGGER.warn(
                                    "[LodGen] Column task failed: {}",
                                    e.getMessage());
                        }
                    }
                }

                executor.shutdownNow();
            } else {
                // ── Single-threaded pass for finer LODs ──────────────
                for (int[] col : columns) {
                    if (stopRequested.get()) return;
                    int count = processColumn(world, model, writer, blockMapper,
                            col[0], col[1], lod);
                    int total = passCount.addAndGet(count);
                    if (count > 0 && total % 200 < count) {
                        HelloTerrainMod.LOGGER.info(
                                "[LodGen] LOD {} progress: {} sections " +
                                "(skipped {} air)",
                                lod, total, skippedAirSections.get());
                    }
                }
            }

            HelloTerrainMod.LOGGER.info(
                    "[LodGen] LOD {} pass complete — {} sections generated, {} skipped " +
                    "(noise={}, real={}, synthetic={})",
                    lod, passCount.get(), skippedAirSections.get(),
                    noiseAccessSections.get(), realDataSections.get(),
                    syntheticDataSections.get());

            // Free cached parents outside the next pass's radius
            if (lod > FINEST_LOD) {
                pruneParentCache(centerX, centerZ, PASS_RADIUS[lod - 1]);
            } else {
                parentCache.clear();
            }
        }

        HelloTerrainMod.LOGGER.info("[LodGen] All LOD passes complete");
    }

    /**
     * Process all Y sections for a single column.
     *
     * <p>Thread-safe — multiple columns can be processed concurrently.
     * Each column samples its own conditioning data and writes results
     * independently to Voxy.
     *
     * @return the number of sections actually generated
     */
    private int processColumn(World world, ProgressiveModelRunner model,
                               VoxySectionWriter writer,
                               VoxyBlockMapper blockMapper,
                               int sx, int sz, int lod) {
        if (stopRequested.get()) return 0;

        // Skip columns where vanilla has loaded real chunks
        if (tryGetLoadedChunk(world, sx, sz) != null) {
            writer.forgetColumn(sx, sz, Y_BASE_SECTION, Y_SECTIONS);
            return 0;
        }

        // Sample conditioning data ONCE per column
        ColumnContext ctx = buildColumnContext(world, sx, sz);

        // Compute Y range from surface heightmap
        float minH = Float.MAX_VALUE, maxH = -Float.MAX_VALUE;
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                float h = ctx.rawHm[lx][lz];
                if (h < minH) minH = h;
                if (h > maxH) maxH = h;
            }
        }

        int minSectionY = Math.floorDiv((int) Math.floor(minH), 16) - SURFACE_MARGIN;
        int maxSectionY = Math.floorDiv((int) Math.ceil(maxH), 16) + SURFACE_MARGIN;

        // Log heightmap range for first few columns
        if (diagnosticCount.get() < 3) {
            diagnosticCount.incrementAndGet();
            HelloTerrainMod.LOGGER.info(
                    "[LodGen] Column ({},{}) heightmap: min={} max={} → "
                    + "sectionY=[{},{}] (blocks [{},{}])",
                    sx, sz, minH, maxH,
                    Math.max(minSectionY, Y_BASE_SECTION),
                    Math.min(maxSectionY, Y_BASE_SECTION + Y_SECTIONS - 1),
                    Math.max(minSectionY, Y_BASE_SECTION) * 16,
                    (Math.min(maxSectionY, Y_BASE_SECTION + Y_SECTIONS - 1) + 1) * 16 - 1);
        }

        // Clamp to the valid Y range
        minSectionY = Math.max(minSectionY, Y_BASE_SECTION);
        maxSectionY = Math.min(maxSectionY, Y_BASE_SECTION + Y_SECTIONS - 1);

        int generated = 0;
        for (int sy = minSectionY; sy <= maxSectionY; sy++) {
            if (stopRequested.get()) break;

            long lodKey = sectionKey(sx, sy, sz, lod);
            if (generatedSections.contains(lodKey)) continue;

            try {
                inferAndPushSection(world, model, writer, blockMapper,
                        ctx, sx, sy, sz, lod);
                generatedSections.add(lodKey);
                generated++;
            } catch (Exception e) {
                HelloTerrainMod.LOGGER.warn(
                        "[LodGen] Failed ({},{},{}) LOD {}: {}",
                        sx, sy, sz, lod, e.getMessage());
            }

            // Cooperatively yield instead of hard-sleeping 5ms per section.
            // The old Thread.sleep(5) added 15+ seconds of pure sleeping
            // per LOD pass (~3000 sections × 5ms each).
            Thread.yield();
            if (stopRequested.get()) break;
        }

        // Count sections we skipped (above/below surface)
        int generatedRange = maxSectionY - minSectionY + 1;
        skippedAirSections.addAndGet(Y_SECTIONS - generatedRange);

        return generated;
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
    private record ColumnContext(
        float[][] rawHm,       // [16][16] surface heightmap in block Y
        int[][]   biomeIdx,    // [16][16] biome indices
        float[][] hp5,         // [5][256] height-planes (row-major)
        float[][] r6           // [6][256] router6 values (row-major)
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

        if (noiseAccess != null) {
            // *** PRIMARY PATH: Real noise data at any coordinate ***
            // sampleFromNoise() now returns rawHm inside AnchorInputs — no second
            // sampleHeightmap() call needed (eliminates 256 duplicate getHeight() calls).
            AnchorSampler.AnchorInputs anchor =
                    AnchorSampler.sampleFromNoise(noiseAccess, sectionX, sectionZ);
            rawHm    = anchor.rawHm();
            biomeIdx = anchor.biomeIdx();
            hp5      = anchor.heightPlanes5();
            r6       = anchor.router6();
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

        return new ColumnContext(rawHm, biomeIdx, hp5, r6);
    }

    /**
     * Run a single inference pass for one section at a given LOD,
     * then push the result to Voxy immediately.
     *
     * <p>For the coarsest LOD (4), uses a heightmap-based initial parent.
     * For finer LODs, uses the cached coarsened output from the previous
     * LOD pass as the parent.
     *
     * @param ctx pre-sampled column conditioning (heightmap, biome, router6)
     */
    private void inferAndPushSection(World world,
                                      ProgressiveModelRunner model,
                                      VoxySectionWriter writer,
                                      VoxyBlockMapper blockMapper,
                                      ColumnContext ctx,
                                      int sectionX, int sectionY,
                                      int sectionZ, int lod)
            throws Exception {

        // 0-based y index: sectionY -4 → 0, -3 → 1, ..., 11 → 15
        // Matches training extraction: y_index = world_y - Y_BASE_SECTION
        // The ONNX model has clamp(0, 23) built in for safety.
        int yIndex = sectionY - Y_BASE_SECTION;

        // Conditioning already sampled per-column in ColumnContext
        int[][]   biomeIdx = ctx.biomeIdx();
        float[][] hp5      = ctx.hp5();

        // Run the full progressive chain: init→LOD4→3→2→1 (8³ upsampled to 16³)
        // ProgressiveModelRunner manages the 4-stage parent chain internally.
        InferenceResult result;
        try {
            result = model.generate(hp5, biomeIdx, yIndex);
        } catch (ai.djl.translate.TranslateException e) {
            throw new RuntimeException("ProgressiveModelRunner inference failed at (" +
                    sectionX + "," + sectionY + "," + sectionZ + ")", e);
        }

        // ---- Heightmap clipping ----
        // Force air above the surface to compensate for undertrained models
        // that predict all-solid.  The model's air mask is overridden for
        // voxels whose world Y >= surface heightmap at that (x,z) column.
        if (HEIGHTMAP_CLIP && ctx.rawHm() != null) {
            float[][][][][] mask = result.airMask(); // [1][1][Y][Z][X]
            int clipped = 0;
            for (int lx = 0; lx < 16; lx++) {
                for (int lz = 0; lz < 16; lz++) {
                    float surfaceY = ctx.rawHm()[lx][lz];
                    for (int ly = 0; ly < 16; ly++) {
                        int worldY = sectionY * 16 + ly;
                        if (worldY >= surfaceY && mask[0][0][ly][lz][lx] > 0f) {
                            mask[0][0][ly][lz][lx] = -1f; // force air
                            clipped++;
                        }
                    }
                }
            }
            if (diagnosticCount.get() < 5 && clipped > 0) {
                HelloTerrainMod.LOGGER.info(
                        "[LodGen] Heightmap clip ({},{},{}): forced {}/4096 voxels to air",
                        sectionX, sectionY, sectionZ, clipped);
            }
        }

        // Diagnostics: sample different Y levels to compare underground vs sky
        boolean shouldDiag = false;
        if (diagnosticCount.get() < 3) {
            // Always log first 3 sections (from first column)
            shouldDiag = true;
        } else if (diagnosticCount.get() < 10) {
            // After first 3, only log extreme Y levels to see above-ground behavior
            // Log Y near surface (sectionY 3-4) and well above (sectionY 8+)
            shouldDiag = sectionY == 4 || sectionY == 8 || sectionY == 11;
        }

        if (shouldDiag) {
            logDiagnostics(result, model.config(), model.vocabulary(), blockMapper,
                    sectionX, sectionY, sectionZ,
                    result.elapsedMs(), lod,
                    yIndex, 0 /* parent not tracked by ProgressiveModelRunner */);
            diagnosticCount.incrementAndGet();
        }

        // Push to Voxy immediately — terrain appears progressively
        // Translate canonical biome IDs → Voxy biome IDs (per-column)
        int[][] biomeVoxyIds = new int[16][16];
        for (int lx = 0; lx < 16; lx++) {
            for (int lz = 0; lz < 16; lz++) {
                biomeVoxyIds[lx][lz] = blockMapper.getVoxyBiomeId(biomeIdx[lx][lz]);
            }
        }
        writer.writeSection(result, model.config().effectiveBlockVocabSize(),
                sectionX, sectionY, sectionZ, biomeVoxyIds);
    }

    // ------------------------------------------------------------------ //
    //  Coarsen & diagnostics
    // ------------------------------------------------------------------ //

    /**
     * Coarsen a 16³ air-mask to an 8³ binary parent occupancy grid.
     *
     * <p>Uses max-pooling over 2×2×2 blocks: if <em>any</em> voxel in the
     * block has air_mask &gt; 0 (model predicts solid), the parent cell is
     * set to 1.0.  This approximates the Voxy mipper's opacity-biased
     * selection for binary occupancy data.
     *
     * @param airMask model output [1][1][16][16][16]; positive = solid
     * @return [8][8][8] binary parent (0.0 = air, 1.0 = solid)
     */
    static float[][][] coarsenToParent(float[][][][][] airMask) {
        float[][][] parent = new float[8][8][8];
        for (int px = 0; px < 8; px++) {
            for (int py = 0; py < 8; py++) {
                for (int pz = 0; pz < 8; pz++) {
                    float maxVal = -Float.MAX_VALUE;
                    for (int dx = 0; dx < 2; dx++)
                        for (int dy = 0; dy < 2; dy++)
                            for (int dz = 0; dz < 2; dz++) {
                                float v = airMask[0][0]
                                        [px * 2 + dx][py * 2 + dy][pz * 2 + dz];
                                if (v > maxVal) maxVal = v;
                            }
                    parent[px][py][pz] = maxVal > 0f ? 1.0f : 0.0f;
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
     * Remove cached parents outside the given radius from center.
     */
    private void pruneParentCache(int centerX, int centerZ, int keepRadius) {
        parentCache.entrySet().removeIf(e -> {
            long key = e.getKey();
            int x = (short) ((key >> 32) & 0xFFFF);
            int z = (short) (key & 0xFFFF);
            return Math.abs(x - centerX) > keepRadius
                || Math.abs(z - centerZ) > keepRadius;
        });
        HelloTerrainMod.LOGGER.debug(
                "[LodGen] Parent cache pruned to {} entries (keepRadius={})",
                parentCache.size(), keepRadius);
    }

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
        float[][][][][] mask   = result.airMask();
        float[][][][][] logits = result.blockLogits();
        int vocabSize = config.effectiveBlockVocabSize();

        // Air mask stats
        int maskPositive = 0;
        float maskMin = Float.MAX_VALUE, maskMax = -Float.MAX_VALUE;
        for (int d0 = 0; d0 < 16; d0++)
            for (int d1 = 0; d1 < 16; d1++)
                for (int d2 = 0; d2 < 16; d2++) {
                    float v = mask[0][0][d0][d1][d2];
                    if (v > 0) maskPositive++;
                    if (v < maskMin) maskMin = v;
                    if (v > maskMax) maskMax = v;
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
                "[LodGen] DIAG LOD {} ({},{},{}): " +
                "yIdx={} parent={}/512 solid | " +
                "air_mask solid={}/4096 range=[{},{}] | " +
                "logit range=[{},{}] | " +
                "center: idx={} '{}' voxyId={} | " +
                "{}ms",
                lod, sectionX, sectionY, sectionZ,
                yIndex, parentSolid,
                maskPositive, maskMin, maskMax,
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

    private long sectionKey(int x, int y, int z, int lod) {
        return ((long) lod << 48) | ((long) (x & 0xFFFF) << 32)
                | ((long) (y & 0xFFFF) << 16) | (z & 0xFFFFL);
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
