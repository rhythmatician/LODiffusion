package com.rhythmatician.lodiffusion.voxy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.HelloTerrainMod;
import com.rhythmatician.lodiffusion.onnx.UnifiedModelRunner;
import com.rhythmatician.lodiffusion.onnx.UnifiedModelRunner.InferenceResult;

import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

/**
 * Background service that proactively generates LOD sections around the player
 * and pushes them into Voxy for distant rendering.
 *
 * <h3>Architecture</h3>
 * <ol>
 *   <li>On world join, the service starts a worker thread.</li>
 *   <li>Player position is updated each tick from the client thread.</li>
 *   <li>The worker generates sections in priority order:
 *       coarsest LODs (LOD 4) first for distant areas,
 *       then progressively finer LODs for closer areas.</li>
 *   <li>Each section is pushed to Voxy via {@link VoxySectionWriter}.</li>
 * </ol>
 *
 * <p>The ONNX model's {@code x_lod} input (1–4) controls the prediction
 * granularity.  Higher LOD tokens produce coarser but faster predictions
 * that give Voxy something to render at distance before fine detail arrives.
 */
public final class LodGenerationService {

    private static final int COARSEST_LOD = 4;
    private static final int FINEST_LOD = 1;

    /** Radius in sections for each LOD level (sections = 16-block units). */
    private static final int[] LOD_RADIUS = {
        0,   // unused (index 0)
        4,   // LOD 1: 4 sections = 64 blocks
        8,   // LOD 2: 8 sections = 128 blocks
        16,  // LOD 3: 16 sections = 256 blocks
        32   // LOD 4: 32 sections = 512 blocks
    };

    /** How many sections of Y range to generate (from y=-64 upward). */
    private static final int Y_SECTIONS = 16;  // y sections -4..11 → blocks -64..191
    private static final int Y_BASE_SECTION = -4;  // start at y=-64

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private final AtomicBoolean positionReady = new AtomicBoolean(false);
    private volatile Thread workerThread;

    /** Updated each tick from the client thread. */
    private volatile int playerSectionX;
    private volatile int playerSectionZ;

    /** Tracks which sections we've already generated to avoid duplicates. */
    private final Set<Long> generatedSections = new HashSet<>();

    // ------------------------------------------------------------------ //
    //  Lifecycle
    // ------------------------------------------------------------------ //

    /**
     * Start the LOD generation service for a given world.
     *
     * @param world the Minecraft world (client-side)
     */
    public void start(World world) {
        if (running.getAndSet(true)) {
            HelloTerrainMod.LOGGER.warn("[LodGen] Service already running");
            return;
        }

        stopRequested.set(false);
        positionReady.set(false);
        generatedSections.clear();

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
            UnifiedModelRunner model = loadModel();
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

            // Wait for the client tick to supply the real player position
            waitForPlayerPosition();
            if (stopRequested.get()) return;

            HelloTerrainMod.LOGGER.info("[LodGen] Starting generation from player section ({}, {})",
                    playerSectionX, playerSectionZ);

            // Main generation loop: coarsest LODs first, then refine
            generateProgressiveLods(model, writer, blockMapper);

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
     * Generate LOD sections progressively: LOD 4 (coarsest) first, then
     * LOD 3, 2, 1.  Within each LOD level, generates in a spiral outward
     * from the player.
     */
    private void generateProgressiveLods(UnifiedModelRunner model,
                                          VoxySectionWriter writer,
                                          VoxyBlockMapper blockMapper) {
        for (int lod = COARSEST_LOD; lod >= FINEST_LOD; lod--) {
            if (stopRequested.get()) return;

            int radius = LOD_RADIUS[lod];
            int centerX = playerSectionX;
            int centerZ = playerSectionZ;

            HelloTerrainMod.LOGGER.info("[LodGen] Starting LOD {} pass — radius={} sections around ({}, {})",
                    lod, radius, centerX, centerZ);

            // Build section list sorted by distance from player (closest first within each ring)
            List<int[]> sections = buildSpiralSections(centerX, centerZ, radius);
            int generated = 0;

            for (int[] sec : sections) {
                if (stopRequested.get()) return;

                int sx = sec[0];
                int sz = sec[1];

                // Generate for each Y slice
                for (int sy = Y_BASE_SECTION; sy < Y_BASE_SECTION + Y_SECTIONS; sy++) {
                    long key = sectionKey(sx, sy, sz, lod);
                    if (generatedSections.contains(key)) continue;

                    try {
                        generateAndPushSection(model, writer, blockMapper, sx, sy, sz, lod);
                        generatedSections.add(key);
                        generated++;

                        if (generated % 50 == 0) {
                            HelloTerrainMod.LOGGER.info("[LodGen] LOD {} progress: {} sections generated",
                                    lod, generated);
                        }
                    } catch (Exception e) {
                        HelloTerrainMod.LOGGER.warn("[LodGen] Failed section ({},{},{}) LOD {}: {}",
                                sx, sy, sz, lod, e.getMessage());
                    }

                    // Small pause to avoid overwhelming the system
                    try {
                        Thread.sleep(10);
                    } catch (InterruptedException e) {
                        return;
                    }
                }
            }

            HelloTerrainMod.LOGGER.info("[LodGen] LOD {} pass complete — {} sections generated", lod, generated);
        }

        HelloTerrainMod.LOGGER.info("[LodGen] All LOD passes complete");
    }

    /** Counter for detailed diagnostics (first N sections). */
    private int diagnosticCount = 0;

    /**
     * Generate a single 16³ section via ONNX and push to Voxy.
     */
    private void generateAndPushSection(UnifiedModelRunner model,
                                         VoxySectionWriter writer,
                                         VoxyBlockMapper blockMapper,
                                         int sectionX, int sectionY, int sectionZ,
                                         int lodLevel) throws Exception {
        // Build inputs — position-dependent so each section gets unique conditioning
        float[][] rawHeightmap = buildHeightmap(sectionX, sectionZ);
        float[][] normalizedHeightmap = minMaxNormalize(rawHeightmap);
        float avgHeight = averageHeight(rawHeightmap);
        float[][][] parentOccupancy = buildParentOccupancy(sectionY, avgHeight);
        float[][][] biomeOneHot = buildDummyBiomeOneHot(model.config().effectiveBiomeVocabSize());

        // Run inference
        InferenceResult result = model.generate(parentOccupancy, biomeOneHot, normalizedHeightmap, lodLevel);

        boolean detailed = diagnosticCount < 5;

        // Diagnostic: log model output statistics for first few sections
        if (detailed) {
            float[][][][][] mask = result.airMask();
            float[][][][][] logits = result.blockLogits();
            int vocabSize = model.config().effectiveBlockVocabSize();

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

            // Block logit stats: check range across vocab & space
            float logitMin = Float.MAX_VALUE, logitMax = -Float.MAX_VALUE;
            for (int b = 0; b < Math.min(vocabSize, logits[0].length); b++)
                for (int d0 = 0; d0 < 16; d0++)
                    for (int d1 = 0; d1 < 16; d1++)
                        for (int d2 = 0; d2 < 16; d2++) {
                            float v = logits[0][b][d0][d1][d2];
                            if (v < logitMin) logitMin = v;
                            if (v > logitMax) logitMax = v;
                        }

            // Top predicted block index at center voxel (8,8,8)
            int centerBest = 0;
            float centerBestVal = logits[0][0][8][8][8];
            for (int b = 1; b < Math.min(vocabSize, logits[0].length); b++) {
                float v = logits[0][b][8][8][8];
                if (v > centerBestVal) { centerBestVal = v; centerBest = b; }
            }

            // Check what VoxyBlockMapper returns for that index
            int voxyId = blockMapper.getVoxyBlockId(centerBest);
            String blockName = centerBest < model.vocabulary().size()
                    ? model.vocabulary().getName(centerBest) : "???";

            HelloTerrainMod.LOGGER.info(
                    "[LodGen] DIAG section ({},{},{}) LOD {}: " +
                    "air_mask positive={}/4096 range=[{},{}] | " +
                    "logit range=[{},{}] | " +
                    "center prediction: idx={} name='{}' voxyId={} | " +
                    "inference {}ms",
                    sectionX, sectionY, sectionZ, lodLevel,
                    maskPositive, maskMin, maskMax,
                    logitMin, logitMax,
                    centerBest, blockName, voxyId,
                    result.elapsedMs());
            diagnosticCount++;
        }

        // Push to Voxy
        int biomeVoxyId = blockMapper.defaultBiomeVoxyId();
        writer.writeSection(result, model.config().effectiveBlockVocabSize(),
                sectionX, sectionY, sectionZ, biomeVoxyId);

        HelloTerrainMod.LOGGER.debug("[LodGen] Generated section ({},{},{}) LOD {} in {}ms",
                sectionX, sectionY, sectionZ, lodLevel, result.elapsedMs());
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

    /**
     * Per-patch min-max normalise a raw heightmap to [0, 1],
     * matching the training data preparation.
     */
    private float[][] minMaxNormalize(float[][] raw) {
        float min = Float.MAX_VALUE, max = -Float.MAX_VALUE;
        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++) {
                if (raw[x][z] < min) min = raw[x][z];
                if (raw[x][z] > max) max = raw[x][z];
            }
        float range = Math.max(max - min, 1e-6f);
        float[][] norm = new float[16][16];
        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++)
                norm[x][z] = (raw[x][z] - min) / range;
        return norm;
    }

    /**
     * Compute the average raw height across a heightmap.
     */
    private float averageHeight(float[][] raw) {
        float sum = 0;
        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++)
                sum += raw[x][z];
        return sum / 256f;
    }

    /**
     * Build parent occupancy [8][8][8] based on how this section's Y
     * range relates to the average terrain height.
     *
     * <p>Each cell in the 8³ grid represents a 2×2×2 block cube.
     * Cells below the estimated surface are solid (1.0), cells above
     * are air (0.0), with a smooth transition near the surface.
     *
     * @param sectionY   Voxy section Y coordinate
     * @param avgHeight  average terrain height from the heightmap (block Y)
     */
    private float[][][] buildParentOccupancy(int sectionY, float avgHeight) {
        float[][][] occ = new float[8][8][8];
        float sectionBaseY = sectionY * 16f;

        for (int cx = 0; cx < 8; cx++) {
            for (int cy = 0; cy < 8; cy++) {
                for (int cz = 0; cz < 8; cz++) {
                    // World Y at the center of this 2-block-tall cell
                    float cellY = sectionBaseY + cy * 2f + 1f;

                    if (cellY < avgHeight - 8f) {
                        // Well below surface — fully solid
                        occ[cx][cy][cz] = 1.0f;
                    } else if (cellY > avgHeight + 4f) {
                        // Well above surface — air
                        occ[cx][cy][cz] = 0.0f;
                    } else {
                        // Transition zone: linear falloff
                        occ[cx][cy][cz] = Math.max(0f,
                                1.0f - (cellY - (avgHeight - 8f)) / 12f);
                    }
                }
            }
        }
        return occ;
    }

    /**
     * Build dummy biome one-hot: plains (biome index 1).
     */
    private float[][][] buildDummyBiomeOneHot(int biomeVocabSize) {
        float[][][] oneHot = new float[biomeVocabSize][16][16];
        int plainsIdx = Math.min(1, biomeVocabSize - 1);
        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++)
                oneHot[plainsIdx][x][z] = 1.0f;
        return oneHot;
    }

    // ------------------------------------------------------------------ //
    //  Section spiral ordering
    // ------------------------------------------------------------------ //

    /**
     * Build a list of (x, z) section coordinates in a spiral pattern
     * from the center outward, limited to the given radius.
     */
    private List<int[]> buildSpiralSections(int centerX, int centerZ, int radius) {
        List<int[]> sections = new ArrayList<>();
        for (int dx = -radius; dx <= radius; dx++) {
            for (int dz = -radius; dz <= radius; dz++) {
                sections.add(new int[]{centerX + dx, centerZ + dz});
            }
        }
        // Sort by distance from center (Manhattan distance for simplicity)
        sections.sort(Comparator.comparingInt(s ->
                Math.abs(s[0] - centerX) + Math.abs(s[1] - centerZ)));
        return sections;
    }

    // ------------------------------------------------------------------ //
    //  Helpers
    // ------------------------------------------------------------------ //

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
            try { Thread.sleep(100); } catch (InterruptedException e) { return; }
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
    private UnifiedModelRunner loadModel() {
        try {
            java.nio.file.Path onnxPath = Config.modelPath();
            java.nio.file.Path configPath = onnxPath.getParent().resolve("model_config.json");
            return UnifiedModelRunner.load(onnxPath, configPath);
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[LodGen] Model load failed: {}", e.getMessage(), e);
            return null;
        }
    }
}
