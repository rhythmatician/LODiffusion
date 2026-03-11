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
import com.rhythmatician.lodiffusion.onnx.OctreeModelRunner;
import java.util.HashMap;

import net.minecraft.registry.Registry;
import net.minecraft.registry.RegistryKeys;
import net.minecraft.server.MinecraftServer;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;
import net.minecraft.world.biome.Biome;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkStatus;

/**
 * Background service that generates terrain around the player using the
 * 3-model octree pipeline and pushes results into Voxy for distant rendering.
 *
 * <h3>Architecture — Octree Pipeline</h3>
 * <p>Sections are generated breadth-first using three ONNX models:
 * <ol>
 *   <li><b>L4 (init)</b> — root sections, no parent context. Parallelised
 *       across {@code STAGE_0_PARALLELISM} workers.</li>
 *   <li><b>L3-L1 (refine)</b> — one shared model called with a {@code level}
 *       input. Single worker per level.</li>
 *   <li><b>L0 (leaf)</b> — 32³ block-resolution sections. Results are written
 *       to Voxy as 8 native 16³ sections. Single worker.</li>
 * </ol>
 * Sections are prioritised by Manhattan distance from the player.
 * The pipeline is driven by an {@link OctreeQueue} with one priority-queue
 * per LOD level (0-4).  After inference produces an occupancy mask, occupied
 * octants spawn child tasks at the next finer level.
 */
public final class LodGenerationService {

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
     * When true, force argmax to air (class 0) for voxels above the
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
            Config.getInt("generationRadius", 32);

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
            OctreeModelRunner model = loadModel();
            if (model == null) {
                // ── Heightmap fallback path ──────────────────────────────
                HelloTerrainMod.LOGGER.info(
                        "[LodGen] No ONNX models found — using heightmap fallback generator");

                Object voxyMapper = VoxyCompat.getMapper(worldEngine);
                Registry<Biome> biomeRegistry =
                        world.getRegistryManager().getOrThrow(RegistryKeys.BIOME);
                HeightmapFallbackGenerator.FallbackBlockIds fallbackBlocks =
                        HeightmapFallbackGenerator.resolveBlockIds(voxyMapper);
                int[] fallbackBiomeMappings =
                        HeightmapFallbackGenerator.resolveBiomeMappings(voxyMapper, biomeRegistry);

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
            Registry<Biome> biomeRegistry =
                    world.getRegistryManager().getOrThrow(RegistryKeys.BIOME);
            VoxyBlockMapper blockMapper = VoxyBlockMapper.build(model.vocabulary(), voxyMapper, biomeRegistry);
            VoxySectionWriter writer = new VoxySectionWriter(worldEngine, blockMapper);

            HelloTerrainMod.LOGGER.info("[LodGen] Ready — waiting for player position " +
                    "(vocab={}, biomeVoxyId={})", model.vocabulary().size(), blockMapper.defaultBiomeVoxyId());

            // Wait for the client tick to supply the real player position
            waitForPlayerPosition();
            if (stopRequested.get()) return;

            HelloTerrainMod.LOGGER.info("[LodGen] Starting generation from player section ({}, {})",
                    playerSectionX, playerSectionZ);

            // Run the octree pipeline
            runOctreePipeline(world, model, writer, blockMapper);

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
     * Load all three octree ONNX models from the configured model directory.
     *
     * @return the loaded runner, or {@code null} if models are absent / failed to load
     */
    private OctreeModelRunner loadModel() {
        try {
            java.nio.file.Path modelDir = Config.modelDir();
            HelloTerrainMod.LOGGER.info("[LodGen] Loading octree models from {}...", modelDir);
            return OctreeModelRunner.loadAll(modelDir);
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[LodGen] Model load failed: {}", e.getMessage(), e);
            return null;
        }
    }

    // ------------------------------------------------------------------ //
    //  Octree pipeline
    // ------------------------------------------------------------------ //

    /**
     * Build an {@link OctreeColumnContext} for an octree section at
     * {@code level} with WorldSection coordinates {@code (wsX, wsY, wsZ)}.
     *
     * <p>Samples heightmap and biome data at 32x32 grid points whose
     * sub-voxel step-size equals {@code 1 << level} blocks.  Uses
     * {@link WorldNoiseAccess} if available, otherwise falls back to
     * the synthetic sine-wave heightmap.
     */
    private OctreeColumnContext buildOctreeColumnContext(int level, int wsX, int wsY, int wsZ) {
        int blockStep = 1 << level;                   // blocks per octree voxel
        float[][] rawHm = new float[32][32];
        int[][]   biomeIdx = new int[32][32];

        // Per-chunk cache for this call (chunk key -> float[2][16][16])
        HashMap<Long, float[][][]> chunkCache     = new HashMap<>();
        HashMap<Long, String[][]>  biomeNameCache = new HashMap<>();

        for (int cx = 0; cx < 32; cx++) {
            for (int cz = 0; cz < 32; cz++) {
                // Block coordinate of cell center
                int bx = wsX * 32 * blockStep + cx * blockStep + blockStep / 2;
                int bz = wsZ * 32 * blockStep + cz * blockStep + blockStep / 2;

                int chunkX = bx >> 4;
                int chunkZ = bz >> 4;
                int lx = bx & 15;
                int lz = bz & 15;

                long chunkKey = (long) chunkX << 32 | (chunkZ & 0xFFFFFFFFL);

                if (noiseAccess != null) {
                    float[][][] heights = chunkCache.computeIfAbsent(chunkKey,
                            k -> noiseAccess.sampleBothHeightmaps(chunkX, chunkZ));
                    rawHm[cz][cx] = heights[0][lx][lz];

                    String[][] names = biomeNameCache.computeIfAbsent(chunkKey,
                            k -> noiseAccess.sampleBiomeNames(chunkX, chunkZ, heights[0]));
                    biomeIdx[cz][cx] = BiomeMapping.toCanonicalId(names[lx][lz]);
                } else {
                    // Synthetic fallback (sine-wave height)
                    float h = buildSingleHeight(bx, bz);
                    rawHm[cz][cx] = h;
                    biomeIdx[cz][cx] = 1; // plains default
                }
            }
        }

        // Compute 5-plane heightmap from 32x32 rawHm (mirrors AnchorSampler.computeHeightPlanes)
        float[][][] heightmap5 = computeOctreeHeightPlanes(rawHm);

        return new OctreeColumnContext(heightmap5, biomeIdx, rawHm);
    }

    /**
     * Compute the 5-plane height tensor for a 32x32 heightmap.
     * Mirrors {@code AnchorSampler.computeHeightPlanes} extended to 32x32.
     */
    private static float[][][] computeOctreeHeightPlanes(float[][] rawHm) {
        final float HEIGHT_RANGE = 320f;
        final float SEA_LEVEL_PLANE    = 62f;
        float[][][] planes = new float[5][32][32];

        float[][] surfNorm = new float[32][32];
        float[][] slopeX   = new float[32][32];
        float[][] slopeZ   = new float[32][32];

        for (int r = 0; r < 32; r++) {
            for (int c = 0; c < 32; c++) {
                float h = rawHm[r][c];
                surfNorm[r][c] = h / HEIGHT_RANGE;
                planes[0][r][c] = surfNorm[r][c];                         // surface
                planes[1][r][c] = Math.min(h, SEA_LEVEL_PLANE) / HEIGHT_RANGE; // ocean_floor approx
            }
        }
        for (int r = 0; r < 32; r++) {
            for (int c = 0; c < 32; c++) {
                if (c == 0)     slopeX[r][c] = surfNorm[r][1] - surfNorm[r][0];
                else if (c==31) slopeX[r][c] = surfNorm[r][31] - surfNorm[r][30];
                else            slopeX[r][c] = (surfNorm[r][c+1] - surfNorm[r][c-1]) / 2f;
                planes[2][r][c] = slopeX[r][c];

                if (r == 0)     slopeZ[r][c] = surfNorm[1][c] - surfNorm[0][c];
                else if (r==31) slopeZ[r][c] = surfNorm[31][c] - surfNorm[30][c];
                else            slopeZ[r][c] = (surfNorm[r+1][c] - surfNorm[r-1][c]) / 2f;
                planes[3][r][c] = slopeZ[r][c];
            }
        }
        for (int r = 0; r < 32; r++) {
            for (int c = 0; c < 32; c++) {
                float dsx = (c == 0)  ? slopeX[r][1] - slopeX[r][0]
                          : (c == 31) ? slopeX[r][31] - slopeX[r][30]
                          : (slopeX[r][c+1] - slopeX[r][c-1]) / 2f;
                float dsz = (r == 0)  ? slopeZ[1][c] - slopeZ[0][c]
                          : (r == 31) ? slopeZ[31][c] - slopeZ[30][c]
                          : (slopeZ[r+1][c] - slopeZ[r-1][c]) / 2f;
                planes[4][r][c] = dsx + dsz;
            }
        }
        return planes;
    }

    /** Synthetic single-block height value. */
    private float buildSingleHeight(int bx, int bz) {
        float h = SEA_LEVEL;
        h += HEIGHT_AMPLITUDE * 0.50f * (float) Math.sin(bx * 0.005 + 1.7)
                                      * (float) Math.cos(bz * 0.007 + 0.3);
        h += HEIGHT_AMPLITUDE * 0.25f * (float) Math.sin(bx * 0.013 + 3.1)
                                      * (float) Math.sin(bz * 0.011 + 2.2);
        h += HEIGHT_AMPLITUDE * 0.12f * (float) Math.cos(bx * 0.037 + 0.9)
                                      * (float) Math.sin(bz * 0.029 + 4.1);
        return Math.max(0f, Math.min(320f, h));
    }

    /**
     * Run the continuous octree pipeline: start level workers, populate L4 roots
     * from the player position, and drive the scheduler until stop is requested.
     */
    private void runOctreePipeline(World world, OctreeModelRunner model,
                                    VoxySectionWriter writer, VoxyBlockMapper blockMapper) {
        OctreeQueue queue = new OctreeQueue();
        this.activeQueue = null; // octree queue is separate type; kept for fallback compat

        // Register column-context builder for child tasks spawned by OctreeQueue
        queue.setColumnContextBuilder((level, task) ->
                buildOctreeColumnContext(level, task.wsX, task.wsY, task.wsZ));

        int numWorkers = STAGE_0_PARALLELISM + 4; // L4 pool + L3 + L2 + L1 + L0
        Thread[] workers = new Thread[numWorkers];
        java.util.concurrent.atomic.AtomicInteger l4Active =
                new java.util.concurrent.atomic.AtomicInteger(STAGE_0_PARALLELISM);

        // L4 worker pool (no parent -> safe to parallelise)
        for (int i = 0; i < STAGE_0_PARALLELISM; i++) {
            final int idx = i;
            workers[i] = new Thread(() -> {
                try {
                    runOctreeLevelWorker(4, queue, model, writer, blockMapper);
                } finally {
                    if (l4Active.decrementAndGet() == 0) {
                        queue.signalLevelComplete(4);
                    }
                }
            }, "LODiffusion-L4-" + idx);
            workers[i].setDaemon(true);
        }

        // L3-L0 workers (single-threaded per level)
        for (int lvl = 3; lvl >= 0; lvl--) {
            final int level = lvl;
            int wIdx = STAGE_0_PARALLELISM + (3 - lvl);
            workers[wIdx] = new Thread(() -> {
                try {
                    runOctreeLevelWorker(level, queue, model, writer, blockMapper);
                } finally {
                    queue.signalLevelComplete(level);
                }
            }, "LODiffusion-L" + lvl);
            workers[wIdx].setDaemon(true);
        }

        for (Thread w : workers) w.start();

        HelloTerrainMod.LOGGER.info(
                "[LodGen] Octree pipeline starting — {} L4 workers, radius={}",
                STAGE_0_PARALLELISM, GENERATION_RADIUS);

        // Root-population loop
        waitForPlayerPosition();
        if (stopRequested.get()) {
            queue.signalPopulationDone();
            return;
        }

        // L4 sections cover 32 L0-sections per axis -> L4 radius = L0_radius / 32
        int l4Radius = Math.max(1, GENERATION_RADIUS / 32);

        int lastCenterX = Integer.MIN_VALUE;
        int lastCenterZ = Integer.MIN_VALUE;

        while (!stopRequested.get()) {
            int px = playerSectionX;
            int pz = playerSectionZ;
            int l4Cx = px >> 5; // L0 section -> L4 section coordinate
            int l4Cz = pz >> 5;

            if (l4Cx != lastCenterX || l4Cz != lastCenterZ) {
                lastCenterX = l4Cx;
                lastCenterZ = l4Cz;

                for (int dx = -l4Radius; dx <= l4Radius; dx++) {
                    for (int dz = -l4Radius; dz <= l4Radius; dz++) {
                        if (stopRequested.get()) break;
                        int wx = l4Cx + dx;
                        int wz = l4Cz + dz;
                        // Y range: fit the Y_SECTIONS range, converted to L4 coordinates
                        int wyMin = (Y_BASE_SECTION >> 5) - 1;
                        int wyMax = ((Y_BASE_SECTION + Y_SECTIONS) >> 5) + 1;
                        for (int wy = wyMin; wy <= wyMax; wy++) {
                            int priority = Math.abs(dx) + Math.abs(dz);
                            OctreeTask root = new OctreeTask(4, wx, wy, wz, -1, priority);
                            root.columnContext = buildOctreeColumnContext(4, wx, wy, wz);
                            queue.enqueueRoot(root);
                        }
                    }
                }
            }

            // Cancel tasks beyond radius
            queue.cancelBeyondRadius(px, pz, GENERATION_RADIUS + CANCEL_MARGIN);

            try {
                Thread.sleep(500);
            } catch (InterruptedException e) {
                break;
            }
        }

        queue.signalPopulationDone();

        // Wait for all workers to drain
        for (Thread w : workers) {
            try {
                w.join(10_000);
            } catch (InterruptedException ignored) {
                break;
            }
        }

        HelloTerrainMod.LOGGER.info(
                "[LodGen] Octree pipeline complete — {} done, {} failed",
                queue.completedCount(), queue.failedCount());
    }

    /**
     * Worker loop for a single octree LOD level.  Drains batches of tasks from
     * the level queue, runs inference, writes L0 results to Voxy, and spawns
     * children via the occupancy mask.
     */
    private void runOctreeLevelWorker(int level, OctreeQueue queue,
                                       OctreeModelRunner model,
                                       VoxySectionWriter writer,
                                       VoxyBlockMapper blockMapper) {
        String tName = Thread.currentThread().getName();
        HelloTerrainMod.LOGGER.info("[LodGen] {} starting", tName);
        int processed = 0;

        while (!stopRequested.get()) {
            List<OctreeTask> batch;
            try {
                batch = queue.drainLevel(level, MAX_BATCH_SIZE, 200, TimeUnit.MILLISECONDS);
            } catch (InterruptedException e) {
                break;
            }

            if (batch.isEmpty()) {
                if (queue.isUpstreamDone(level)) {
                    // Final drain
                    batch = new ArrayList<>();
                    OctreeTask last;
                    while ((last = queue.pollLevel(level)) != null) batch.add(last);
                    if (batch.isEmpty()) break;
                } else {
                    continue;
                }
            }

            List<OctreeTask> claimed = new ArrayList<>(batch.size());
            for (OctreeTask t : batch) {
                if (t.claimForProcessing()) claimed.add(t);
            }
            if (claimed.isEmpty()) continue;

            for (OctreeTask task : claimed) {
                try {
                    processOctreeTask(task, queue, model, writer, blockMapper);
                    processed++;
                } catch (Exception e) {
                    task.markFailed(e.getMessage());
                    queue.markFailed();
                    if (!stopRequested.get()) {
                        HelloTerrainMod.LOGGER.warn(
                                "[LodGen] {} task L{}({},{},{}) failed: {}",
                                tName, task.level, task.wsX, task.wsY, task.wsZ,
                                e.getMessage(), e);
                    }
                }
            }

            if (processed % 200 < claimed.size()) {
                HelloTerrainMod.LOGGER.info(
                        "[LodGen] {} progress: {} processed,  queues: {}",
                        tName, processed, queue.queueSizeSummary());
            }
        }

        HelloTerrainMod.LOGGER.info("[LodGen] {} exiting — processed {} tasks", tName, processed);
    }

    /**
     * Process a single octree task: run inference, write to Voxy (L0 only),
     * spawn children via occupancy mask.
     */
    private void processOctreeTask(OctreeTask task, OctreeQueue queue,
                                    OctreeModelRunner model,
                                    VoxySectionWriter writer,
                                    VoxyBlockMapper blockMapper) throws Exception {
        OctreeColumnContext ctx = task.columnContext;
        if (ctx == null) {
            ctx = buildOctreeColumnContext(task.level, task.wsX, task.wsY, task.wsZ);
            task.columnContext = ctx;
        }

        OctreeModelRunner.OctreeOutput output;
        if (task.level == 4) {
            output = model.runInit(ctx, task.wsY);
        } else if (task.level > 0) {
            output = model.runRefine(task.parentContextFlat, ctx, task.wsY, task.level);
        } else {
            output = model.runLeaf(task.parentContextFlat, ctx, task.wsY);
        }

        // Write to Voxy only at the leaf level
        if (task.level == 0) {
            writer.writeOctreeBlockData(
                    output.blockLogits(),
                    ctx.biomeIdx(),
                    output.vocabSize(),
                    task.wsX, task.wsY, task.wsZ);
        }

        // Spawn children for non-leaf levels
        if (task.level > 0) {
            int spawned = queue.spawnChildren(task, output.occMask(), output.blockArgmax());
            if (diagnosticCount.get() < 10) {
                HelloTerrainMod.LOGGER.info(
                        "[LodGen] L{} ({},{},{}) — occMask=0x{} spawned={} {}ms",
                        task.level, task.wsX, task.wsY, task.wsZ,
                        Integer.toHexString(output.occMask() & 0xFF),
                        spawned, output.elapsedMs());
                diagnosticCount.incrementAndGet();
            }
        }

        task.markReady();
        queue.markCompleted();
    }

}
