package com.rhythmatician.lodiffusion.terrain;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.HelloTerrainMod;
import com.rhythmatician.lodiffusion.onnx.BlockVocabulary;
import com.rhythmatician.lodiffusion.onnx.UnifiedModelRunner;
import com.rhythmatician.lodiffusion.onnx.UnifiedModelRunner.InferenceResult;
import com.rhythmatician.lodiffusion.util.PerformanceMonitor;

import net.minecraft.block.BlockState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.Chunk;

/**
 * TerrainGenerator implementation backed by a single unified VoxelTree ONNX model.
 *
 * <p>Loads the model from the configured path plus its sidecar
 * {@code model_config.json}, then delegates to {@link UnifiedModelRunner} for
 * inference and {@link BlockVocabulary} for index → BlockState translation.
 */
public class OnnxTerrainGenerator implements TerrainGenerator {

    private static final Object LOCK = new Object();
    private static volatile UnifiedModelRunner runner;
    private static volatile boolean loadAttempted = false;
    private static volatile String lastError = null;

    // ------------------------------------------------------------------ //
    //  TerrainGenerator
    // ------------------------------------------------------------------ //

    @Override
    @SuppressWarnings("try")
    public void generateChunk(ChunkPos pos, Chunk chunk, long seed) {
        try (var totalTimer = PerformanceMonitor.startTiming(PerformanceMonitor.TOTAL_GENERATION_TIME)) {
            PerformanceMonitor.incrementCounter(PerformanceMonitor.CHUNKS_GENERATED);

            UnifiedModelRunner model = ensureLoaded();

            HelloTerrainMod.LOGGER.debug("[OnnxTerrainGenerator] Generating chunk ({}, {}) with unified model", pos.x, pos.z);

            // 1. Extract inputs from chunk
            int[][] heightmap = extractHeightmapFromChunk(chunk);
            int[][] biomeIds  = extractBiomeIdsFromChunk(chunk);
            int baseY = calculateBaseY(heightmap);

            // 2. Build tensors for the v1 contract
            float[][][] parentOccupancy = buildParentOccupancy(chunk, baseY);
            float[][][] biomeOneHot     = buildBiomeOneHot(biomeIds, model.config().effectiveBiomeVocabSize());
            float[][]   normalizedHeight = buildNormalizedHeightmap(heightmap);

            // 3. Run inference (LOD level 1 = finest detail)
            InferenceResult result = model.generate(parentOccupancy, biomeOneHot, normalizedHeight, 1);

            // 4. Argmax + air mask → final block indices [16][16][16]
            int[][][] blockIndices = decodeOutput(result, model.config().effectiveBlockVocabSize());

            // 5. Place blocks using vocabulary → BlockState lookup
            applyTerrainToChunk(chunk, blockIndices, baseY, model.vocabulary());

            PerformanceMonitor.incrementCounter(PerformanceMonitor.ONNX_INFERENCES);
            HelloTerrainMod.LOGGER.debug("[OnnxTerrainGenerator] Chunk ({}, {}) completed in {}ms",
                    pos.x, pos.z, result.elapsedMs());

        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[OnnxTerrainGenerator] Error generating chunk ({}, {}): {}",
                    pos.x, pos.z, e.getMessage(), e);
            PerformanceMonitor.incrementCounter(PerformanceMonitor.MODEL_ERRORS);
            throw new RuntimeException("ONNX terrain generation failed for chunk ("
                    + pos.x + ", " + pos.z + ")", e);
        }
    }

    // ------------------------------------------------------------------ //
    //  Static readiness API  (used by mixin + command)
    // ------------------------------------------------------------------ //

    /** True when the unified model is loaded and ready for inference. */
    public static boolean isReady() {
        if (runner != null) return true;
        // Don't trigger load – just check file existence
        Path modelDir = Config.modelPath().getParent();
        if (modelDir == null) return false;
        return Files.isRegularFile(Config.modelPath())
            && Files.isRegularFile(modelDir.resolve("model_config.json"));
    }

    /** Diagnostic string for the /lodiffusion status command. */
    public static String getStatusInfo() {
        StringBuilder sb = new StringBuilder("ONNX Terrain Status:\n");
        sb.append("- Model loaded: ").append(runner != null).append('\n');
        sb.append("- Model path:   ").append(Config.modelPath()).append('\n');
        if (runner != null) {
            sb.append("- Contract:     ").append(runner.config().contract()).append('\n');
            sb.append("- Block vocab:  ").append(runner.vocabulary().size()).append('\n');
            sb.append("- Biome vocab:  ").append(runner.config().effectiveBiomeVocabSize()).append('\n');
        }
        if (lastError != null) {
            sb.append("- Last error:   ").append(lastError).append('\n');
        }
        return sb.toString();
    }

    /** Force-reload the model (called by /lodiffusion reload). */
    public static void reload() {
        synchronized (LOCK) {
            if (runner != null) {
                runner.close();
                runner = null;
            }
            loadAttempted = false;
            lastError = null;
        }
    }

    // ------------------------------------------------------------------ //
    //  Model loading
    // ------------------------------------------------------------------ //

    private static UnifiedModelRunner ensureLoaded() {
        UnifiedModelRunner r = runner;
        if (r != null) return r;
        synchronized (LOCK) {
            if (runner != null) return runner;
            if (loadAttempted) {
                throw new IllegalStateException("Model load already failed: " + lastError);
            }
            loadAttempted = true;
            try {
                Path onnxPath = Config.modelPath();
                Path configPath = onnxPath.getParent().resolve("model_config.json");
                runner = UnifiedModelRunner.load(onnxPath, configPath);
                HelloTerrainMod.LOGGER.info("[OnnxTerrainGenerator] Unified model loaded from {}", onnxPath);
                return runner;
            } catch (IOException e) {
                lastError = e.getMessage();
                throw new IllegalStateException("Failed to load unified ONNX model", e);
            }
        }
    }

    // ------------------------------------------------------------------ //
    //  Input building
    // ------------------------------------------------------------------ //

    /** Build 8×8×8 binary occupancy from the chunk around baseY. */
    private float[][][] buildParentOccupancy(Chunk chunk, int baseY) {
        float[][][] occ = new float[8][8][8];
        ChunkPos cp = chunk.getPos();
        int ox = cp.getStartX();
        int oz = cp.getStartZ();
        for (int x = 0; x < 8; x++) {
            for (int y = 0; y < 8; y++) {
                for (int z = 0; z < 8; z++) {
                    // Sample every-other block to cover a 16-wide area at half res
                    BlockPos bp = new BlockPos(ox + x * 2, baseY + y * 2, oz + z * 2);
                    occ[x][y][z] = chunk.getBlockState(bp).isAir() ? 0f : 1f;
                }
            }
        }
        return occ;
    }

    /** Build one-hot biome tensor [biomeVocabSize][16][16]. */
    private float[][][] buildBiomeOneHot(int[][] biomeIds, int biomeVocabSize) {
        float[][][] oneHot = new float[biomeVocabSize][16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int id = Math.max(0, Math.min(biomeVocabSize - 1, biomeIds[x][z]));
                oneHot[id][x][z] = 1f;
            }
        }
        return oneHot;
    }

    /** Build normalised [16][16] heightmap in [0, 1]. */
    private float[][] buildNormalizedHeightmap(int[][] heightmap) {
        float[][] norm = new float[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                // MC world range: -64..319  →  normalise to [0, 1]
                norm[x][z] = (heightmap[x][z] + 64f) / 384f;
            }
        }
        return norm;
    }

    // ------------------------------------------------------------------ //
    //  Output decoding
    // ------------------------------------------------------------------ //

    /**
     * Argmax over block logits then apply air mask.
     *
     * @return block indices [16][16][16] (0 = air)
     */
    private int[][][] decodeOutput(InferenceResult result, int vocabSize) {
        float[][][][][] logits = result.blockLogits();  // [1][N][16][16][16]
        float[][][][][] mask   = result.airMask();      // [1][1][16][16][16]
        int[][][] out = new int[16][16][16];

        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    // Air mask: positive logit → solid
                    if (mask[0][0][x][y][z] <= 0f) {
                        out[x][y][z] = 0; // air
                        continue;
                    }
                    // Argmax over vocab dimension
                    int best = 0;
                    float bestVal = logits[0][0][x][y][z];
                    for (int b = 1; b < vocabSize; b++) {
                        float v = logits[0][b][x][y][z];
                        if (v > bestVal) { bestVal = v; best = b; }
                    }
                    out[x][y][z] = best;
                }
            }
        }
        return out;
    }

    // ------------------------------------------------------------------ //
    //  Chunk I/O helpers
    // ------------------------------------------------------------------ //

    private int[][] extractHeightmapFromChunk(Chunk chunk) {
        int[][] hm = new int[16][16];
        ChunkPos cp = chunk.getPos();
        int ox = cp.getStartX();
        int oz = cp.getStartZ();
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 319; y >= -64; y--) {
                    if (!chunk.getBlockState(new BlockPos(ox + x, y, oz + z)).isAir()) {
                        hm[x][z] = y;
                        break;
                    }
                }
            }
        }
        return hm;
    }

    private int[][] extractBiomeIdsFromChunk(Chunk chunk) {
        int[][] ids = new int[16][16];
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                // MC biome palette is 4×4 resolution; getBiomeForNoiseGen takes
                // biome coordinates (block >> 2), y section, z biome coord.
                var biome = chunk.getBiomeForNoiseGen(x >> 2, 16, z >> 2);
                ids[x][z] = Math.abs(biome.hashCode()) % 256;
            }
        }
        return ids;
    }

    private int calculateBaseY(int[][] heightmap) {
        long sum = 0;
        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++)
                sum += heightmap[x][z];
        return (int) (sum / 256);
    }

    /** Write decoded block indices into the chunk using BlockVocabulary. */
    private void applyTerrainToChunk(Chunk chunk, int[][][] blocks, int baseY, BlockVocabulary vocab) {
        ChunkPos cp = chunk.getPos();
        int ox = cp.getStartX();
        int oz = cp.getStartZ();
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 16; y++) {
                    int worldY = baseY + y;
                    if (worldY < -64 || worldY >= 320) continue;
                    int idx = blocks[x][y][z];
                    BlockState state = vocab.getState(idx);
                    chunk.setBlockState(new BlockPos(ox + x, worldY, oz + z), state, false);
                }
            }
        }
    }
}
