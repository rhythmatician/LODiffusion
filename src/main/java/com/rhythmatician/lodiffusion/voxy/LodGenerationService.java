package com.rhythmatician.lodiffusion.voxy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicBoolean;

import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.HelloTerrainMod;
import com.rhythmatician.lodiffusion.onnx.UnifiedModelRunner;
import com.rhythmatician.lodiffusion.onnx.UnifiedModelRunner.InferenceResult;

import net.minecraft.util.math.BlockPos;
import net.minecraft.world.World;

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

    private final AtomicBoolean running = new AtomicBoolean(false);
    private final AtomicBoolean stopRequested = new AtomicBoolean(false);
    private final AtomicBoolean positionReady = new AtomicBoolean(false);
    private volatile Thread workerThread;

    /** Updated each tick from the client thread. */
    private volatile int playerSectionX;
    private volatile int playerSectionZ;

    /** Tracks which (section, lod) combos we've already generated. */
    private final Set<Long> generatedSections = new HashSet<>();

    /** Coarsened parent cache: posKey → float[8][8][8] from previous LOD pass. */
    private final Map<Long, float[][][]> parentCache = new HashMap<>();

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
        parentCache.clear();

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
                                          UnifiedModelRunner model,
                                          VoxySectionWriter writer,
                                          VoxyBlockMapper blockMapper) {
        int centerX = playerSectionX;
        int centerZ = playerSectionZ;

        HelloTerrainMod.LOGGER.info(
                "[LodGen] Starting progressive generation — " +
                "{} passes (LOD {}→{}) around ({}, {})",
                COARSEST_LOD - FINEST_LOD + 1,
                COARSEST_LOD, FINEST_LOD, centerX, centerZ);

        for (int lod = COARSEST_LOD; lod >= FINEST_LOD; lod--) {
            if (stopRequested.get()) return;

            int radius = PASS_RADIUS[lod];
            List<int[]> columns = buildSpiralSections(centerX, centerZ, radius);
            int passCount = 0;
            diagnosticCount = 0;  // reset per pass

            HelloTerrainMod.LOGGER.info(
                    "[LodGen] LOD {} pass — radius={}, ~{} columns × {} Y = ~{} sections",
                    lod, radius, columns.size(), Y_SECTIONS,
                    columns.size() * Y_SECTIONS);

            for (int[] col : columns) {
                if (stopRequested.get()) return;
                int sx = col[0], sz = col[1];

                for (int sy = Y_BASE_SECTION; sy < Y_BASE_SECTION + Y_SECTIONS; sy++) {
                    long lodKey = sectionKey(sx, sy, sz, lod);
                    if (generatedSections.contains(lodKey)) continue;

                    try {
                        inferAndPushSection(model, writer, blockMapper,
                                sx, sy, sz, lod);
                        generatedSections.add(lodKey);
                        passCount++;

                        if (passCount % 200 == 0) {
                            HelloTerrainMod.LOGGER.info(
                                    "[LodGen] LOD {} progress: {} sections",
                                    lod, passCount);
                        }
                    } catch (Exception e) {
                        HelloTerrainMod.LOGGER.warn(
                                "[LodGen] Failed ({},{},{}) LOD {}: {}",
                                sx, sy, sz, lod, e.getMessage());
                    }

                    try { Thread.sleep(5); }
                    catch (InterruptedException e) { return; }
                }
            }

            HelloTerrainMod.LOGGER.info(
                    "[LodGen] LOD {} pass complete — {} sections",
                    lod, passCount);

            // Free cached parents outside the next pass's radius
            if (lod > FINEST_LOD) {
                pruneParentCache(centerX, centerZ, PASS_RADIUS[lod - 1]);
            } else {
                parentCache.clear();
            }
        }

        HelloTerrainMod.LOGGER.info("[LodGen] All LOD passes complete");
    }

    /** Counter for detailed diagnostics (reset per LOD pass). */
    private int diagnosticCount = 0;

    /**
     * Run a single inference pass for one section at a given LOD,
     * then push the result to Voxy immediately.
     *
     * <p>For the coarsest LOD (4), uses a heightmap-based initial parent.
     * For finer LODs, uses the cached coarsened output from the previous
     * LOD pass as the parent.
     */
    private void inferAndPushSection(UnifiedModelRunner model,
                                      VoxySectionWriter writer,
                                      VoxyBlockMapper blockMapper,
                                      int sectionX, int sectionY,
                                      int sectionZ, int lod)
            throws Exception {

        int yIndex = Math.max(0, Math.min(23, sectionY - Y_BASE_SECTION));
        long posKey = sectionPosKey(sectionX, sectionY, sectionZ);

        // ---- conditioning inputs ----
        float[][] rawHm    = buildHeightmap(sectionX, sectionZ);
        int[][]   biomeIdx = new int[16][16];
        for (int[] row : biomeIdx) java.util.Arrays.fill(row, 1);

        // Parent: cached from previous LOD pass, or heightmap-based seed
        float[][][] parent = (lod == COARSEST_LOD)
                ? buildInitialParent(sectionY, rawHm)
                : parentCache.getOrDefault(
                      posKey, buildInitialParent(sectionY, rawHm));

        // ---- infer ----
        InferenceResult result;
        if (model.isV2()) {
            float[][] hp5 = AnchorSampler.computeHeightPlanes(rawHm);
            float[][] r6  = AnchorSampler.approximateRouter6(biomeIdx, rawHm);
            result = model.generateV2(
                    parent, hp5, r6, biomeIdx, yIndex, lod);
        } else {
            float[][] normHm     = minMaxNormalize(rawHm);
            float[][][] biomeHot = buildDummyBiomeOneHot(
                    model.config().effectiveBiomeVocabSize());
            result = model.generate(parent, biomeHot, normHm, lod);
        }

        // Cache coarsened output for the next (finer) LOD pass
        if (lod > FINEST_LOD) {
            parentCache.put(posKey, coarsenToParent(result.airMask()));
        } else {
            parentCache.remove(posKey);  // free memory
        }

        // Diagnostics (first few sections of each LOD pass)
        if (diagnosticCount < 5) {
            logDiagnostics(result, model, blockMapper,
                    sectionX, sectionY, sectionZ,
                    result.elapsedMs(), lod);
            diagnosticCount++;
        }

        // Push to Voxy immediately — terrain appears progressively
        int biomeVoxyId = blockMapper.defaultBiomeVoxyId();
        writer.writeSection(result, model.config().effectiveBlockVocabSize(),
                sectionX, sectionY, sectionZ, biomeVoxyId);
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
    private float[][][] buildInitialParent(int sectionY, float[][] heightmap) {
        float[][][] parent = new float[8][8][8];
        float baseY = sectionY * 16f;
        for (int px = 0; px < 8; px++) {
            for (int pz = 0; pz < 8; pz++) {
                float h = heightmap[px * 2][pz * 2];
                for (int py = 0; py < 8; py++) {
                    float cellY = baseY + py * 2f + 1f;
                    parent[px][py][pz] = cellY < h ? 1.0f : 0.0f;
                }
            }
        }
        return parent;
    }

    /**
     * Position-only key for the parent cache (no LOD component).
     */
    private long sectionPosKey(int x, int y, int z) {
        return sectionKey(x, y, z, 0);
    }

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
                                 UnifiedModelRunner model,
                                 VoxyBlockMapper blockMapper,
                                 int sectionX, int sectionY, int sectionZ,
                                 long elapsedMs, int lod) {
        float[][][][][] mask   = result.airMask();
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
            if (v > centerBestVal) { centerBestVal = v; centerBest = b; }
        }

        int voxyId = blockMapper.getVoxyBlockId(centerBest);
        String blockName = centerBest < model.vocabulary().size()
                ? model.vocabulary().getName(centerBest) : "???";

        HelloTerrainMod.LOGGER.info(
                "[LodGen] DIAG LOD {} ({},{},{}): " +
                "air_mask solid={}/4096 range=[{},{}] | " +
                "logit range=[{},{}] | " +
                "center: idx={} '{}' voxyId={} | " +
                "{}ms",
                lod, sectionX, sectionY, sectionZ,
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
