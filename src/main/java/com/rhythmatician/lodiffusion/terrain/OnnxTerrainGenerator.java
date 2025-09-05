package com.rhythmatician.lodiffusion.terrain;

import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.HelloTerrainMod;
import com.rhythmatician.lodiffusion.terrain.adapter.AdapterRegistry;
import com.rhythmatician.lodiffusion.terrain.adapter.OnnxAdapter;
import com.rhythmatician.lodiffusion.terrain.infer.ModelManager;
import com.rhythmatician.lodiffusion.util.DebugUtils;
import com.rhythmatician.lodiffusion.util.PerformanceMonitor;

import ai.djl.Model;
import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.translate.NoopTranslator;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.Chunk;

/**
 * TerrainGenerator implementation that uses ONNX models for terrain generation.
 * Delegates to the appropriate adapter based on configuration.
 */
public class OnnxTerrainGenerator implements TerrainGenerator {
    
    // Reference to the progressive terrain generator for delegation
    private static com.rhythmatician.lodiffusion.OnnxTerrainGenerator progressiveGenerator;
    
    @Override
    @SuppressWarnings("try") // Suppress warnings for unused timer variables (they're auto-closed for timing)
    public void generateChunk(ChunkPos pos, Chunk chunk, long seed) {
        try (var totalTimer = PerformanceMonitor.startTiming(PerformanceMonitor.TOTAL_GENERATION_TIME)) {
            PerformanceMonitor.incrementCounter(PerformanceMonitor.CHUNKS_GENERATED);
            
            // Check if progressive models are available - if so, use progressive generation
            if (areProgressiveModelsAvailable()) {
                HelloTerrainMod.LOGGER.debug("[OnnxTerrainGenerator] Using progressive ONNX generation for chunk ({}, {})", pos.x, pos.z);
                generateChunkProgressive(pos, chunk, seed);
                return;
            }
            
            // Fall back to legacy terrain generation
            HelloTerrainMod.LOGGER.debug("[OnnxTerrainGenerator] Using legacy ONNX generation for chunk ({}, {})", pos.x, pos.z);
            generateChunkLegacy(pos, chunk, seed);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[OnnxTerrainGenerator] Error generating terrain for chunk ({}, {}): {}", 
                pos.x, pos.z, e.getMessage(), e);
            PerformanceMonitor.incrementCounter(PerformanceMonitor.MODEL_ERRORS);
            // Don't rethrow - allow fallback behavior
        }
    }
    
    /**
     * Progressive terrain generation using the main OnnxTerrainGenerator.
     */
    private void generateChunkProgressive(ChunkPos pos, Chunk chunk, long seed) throws Exception {
        // Initialize progressive generator if needed
        if (progressiveGenerator == null) {
            progressiveGenerator = new com.rhythmatician.lodiffusion.OnnxTerrainGenerator();
        }
        
        // Extract heightmap and biomes from chunk
        int[][] heightmap = extractHeightmapFromChunk(chunk);
        int[][] biomeIds = extractBiomeIdsFromChunk(chunk);
        
        // Use progressive terrain generator
        HelloTerrainMod.LOGGER.debug("[OnnxTerrainGenerator] Calling progressive terrain generation...");
        int[][][] generatedTerrain = progressiveGenerator.generateTerrainFromHeights(
            heightmap, biomeIds, calculateBaseY(heightmap), pos.x, pos.z
        );
        
        // Apply generated terrain back to chunk
        applyTerrainToChunk(chunk, generatedTerrain, calculateBaseY(heightmap));
        
        PerformanceMonitor.incrementCounter(PerformanceMonitor.ONNX_INFERENCES);
        HelloTerrainMod.LOGGER.debug("[OnnxTerrainGenerator] Successfully generated progressive terrain for chunk ({}, {})", pos.x, pos.z);
    }
    
    /**
     * Legacy terrain generation using single model and adapters.
     */
    @SuppressWarnings("try") // Suppress warnings for unused timer variables (they're auto-closed for timing)
    private void generateChunkLegacy(ChunkPos pos, Chunk chunk, long seed) throws Exception {
        // Get the configured adapter
        String adapterName = Config.adapter();
        OnnxAdapter adapter = AdapterRegistry.getAdapter(adapterName);
        
        if (adapter == null) {
            HelloTerrainMod.LOGGER.error("[OnnxTerrainGenerator] Unknown adapter: {}", adapterName);
            PerformanceMonitor.incrementCounter(PerformanceMonitor.ADAPTER_ERRORS);
            return;
        }
        
        // Get the model
        Model model = ModelManager.getOrLoad();
        if (model == null) {
            HelloTerrainMod.LOGGER.error("[OnnxTerrainGenerator] Failed to load ONNX model");
            PerformanceMonitor.incrementCounter(PerformanceMonitor.MODEL_ERRORS);
            return;
        }
        
        // Validate adapter compatibility
        if (!adapter.isCompatible(model)) {
            HelloTerrainMod.LOGGER.error("[OnnxTerrainGenerator] Adapter {} is not compatible with model", adapterName);
            PerformanceMonitor.incrementCounter(PerformanceMonitor.ADAPTER_ERRORS);
            return;
        }
        
        // Run inference
        try (NDManager manager = NDManager.newBaseManager();
             Predictor<NDList, NDList> predictor = model.newPredictor(new NoopTranslator())) {
            
            // Extract input from chunk using adapter
            NDArray input;
            try (var extractTimer = PerformanceMonitor.startTiming(PerformanceMonitor.EXTRACT_INPUT_TIME)) {
                input = adapter.extractInput(chunk, pos, seed, manager);
            }
            
            if (input == null) {
                HelloTerrainMod.LOGGER.warn("[OnnxTerrainGenerator] Failed to extract input for chunk ({}, {})", pos.x, pos.z);
                PerformanceMonitor.incrementCounter(PerformanceMonitor.ADAPTER_ERRORS);
                return;
            }
            
            DebugUtils.logTensorSummary(input, "input");
            DebugUtils.dumpTensor(input, "input", pos);
            
            // Run model inference
            NDList outputList;
            try (var inferenceTimer = PerformanceMonitor.startTiming(PerformanceMonitor.MODEL_INFERENCE_TIME)) {
                NDList inputList = new NDList(input);
                outputList = predictor.predict(inputList);
            }
            
            if (outputList.isEmpty()) {
                HelloTerrainMod.LOGGER.warn("[OnnxTerrainGenerator] Model returned empty output for chunk ({}, {})", pos.x, pos.z);
                PerformanceMonitor.incrementCounter(PerformanceMonitor.MODEL_ERRORS);
                return;
            }
            
            NDArray output = outputList.get(0);
            DebugUtils.logTensorSummary(output, "output");
            DebugUtils.dumpTensor(output, "output", pos);
            
            // Apply output to chunk using adapter
            try (var applyTimer = PerformanceMonitor.startTiming(PerformanceMonitor.APPLY_OUTPUT_TIME)) {
                adapter.applyOutput(chunk, pos, output, manager);
            }
            
            PerformanceMonitor.incrementCounter(PerformanceMonitor.ONNX_INFERENCES);
            HelloTerrainMod.LOGGER.debug("[OnnxTerrainGenerator] Successfully generated terrain for chunk ({}, {}) using adapter {}", 
                pos.x, pos.z, adapterName);
        }
    }
    
    /**
     * Check if ONNX terrain generation is ready to use.
     * @return true if model and adapter are available, false otherwise
     */
    public static boolean isReady() {
        try {
            String adapterName = Config.adapter();
            
            // Check for progressive models first (preferred)
            if (areProgressiveModelsAvailable()) {
                return AdapterRegistry.hasAdapter(adapterName);
            }
            
            // Fall back to legacy single model check
            return ModelManager.isAvailable() && 
                   AdapterRegistry.hasAdapter(adapterName);
        } catch (Exception e) {
            return false;
        }
    }
    
    /**
     * Check if progressive models are available by checking the main generator.
     * @return true if the main ONNX terrain generator has progressive models available
     */
    // Static probe cache holder for progressive model bridge
    private static class ProgressiveProbeCache {
        private static volatile boolean cachedProgressiveAvailable = false;
        private static volatile long lastProgressiveProbeMs = 0L;
    }

    private static boolean areProgressiveModelsAvailable() {
        HelloTerrainMod.LOGGER.info("[Terrain OnnxTerrainGenerator] areProgressiveModelsAvailable() called - cached={}, lastProbe={}ms ago", 
            ProgressiveProbeCache.cachedProgressiveAvailable, 
            System.currentTimeMillis() - ProgressiveProbeCache.lastProgressiveProbeMs);
            
        if (ProgressiveProbeCache.cachedProgressiveAvailable) {
            HelloTerrainMod.LOGGER.info("[Terrain OnnxTerrainGenerator] Using cached result: true");
            return true;
        }
        long now = System.currentTimeMillis();
        if (now - ProgressiveProbeCache.lastProgressiveProbeMs < 2000) { // throttle probes (2s)
            HelloTerrainMod.LOGGER.info("[Terrain OnnxTerrainGenerator] Throttling probe ({}ms since last)", 
                now - ProgressiveProbeCache.lastProgressiveProbeMs);
            return false; // still probing / not yet confirmed
        }
        ProgressiveProbeCache.lastProgressiveProbeMs = now;
        try {
            HelloTerrainMod.LOGGER.info("[Terrain OnnxTerrainGenerator] Bridge probe: acquiring main generator instance...");
            com.rhythmatician.lodiffusion.OnnxTerrainGenerator main = com.rhythmatician.lodiffusion.OnnxTerrainGenerator.getInstance();
            if (main == null) {
                HelloTerrainMod.LOGGER.warn("[Terrain OnnxTerrainGenerator] Bridge probe: main generator is null (not constructed yet)");
                return false;
            }

            // First, trust its public availability flag.
            boolean publicAvailable = main.isAvailable();
            HelloTerrainMod.LOGGER.info("[Terrain OnnxTerrainGenerator] Bridge probe: main.isAvailable() => {}", publicAvailable);

            // Deep inspection via reflection (safe): check the four progressive model fields are non-null.
            boolean progressiveFieldsNonNull = false;
            try {
                java.lang.reflect.Field f4 = main.getClass().getDeclaredField("modelLod4to3");
                java.lang.reflect.Field f3 = main.getClass().getDeclaredField("modelLod3to2");
                java.lang.reflect.Field f2 = main.getClass().getDeclaredField("modelLod2to1");
                java.lang.reflect.Field f1 = main.getClass().getDeclaredField("modelLod1to0");
                f4.setAccessible(true); f3.setAccessible(true); f2.setAccessible(true); f1.setAccessible(true);
                Object m4 = f4.get(main);
                Object m3 = f3.get(main);
                Object m2 = f2.get(main);
                Object m1 = f1.get(main);
                progressiveFieldsNonNull = (m4 != null && m3 != null && m2 != null && m1 != null);
                HelloTerrainMod.LOGGER.info("[Terrain OnnxTerrainGenerator] Bridge probe: progressive field presence: L4to3={} L3to2={} L2to1={} L1to0={}",
                        m4 != null, m3 != null, m2 != null, m1 != null);
            } catch (NoSuchFieldException rf) {
                HelloTerrainMod.LOGGER.warn("[Terrain OnnxTerrainGenerator] Bridge probe: reflection failed (field missing) {}", rf.getMessage());
            } catch (Throwable t) {
                HelloTerrainMod.LOGGER.error("[Terrain OnnxTerrainGenerator] Bridge probe: reflection error {}", t.getMessage());
            }

            boolean decided = publicAvailable || progressiveFieldsNonNull;
            HelloTerrainMod.LOGGER.info("[Terrain OnnxTerrainGenerator] Bridge probe: decided progressiveAvailable={} (publicAvailable={}, fieldsNonNull={})",
                    decided, publicAvailable, progressiveFieldsNonNull);

            if (decided) ProgressiveProbeCache.cachedProgressiveAvailable = true;
            return decided;
        } catch (Throwable t) {
            HelloTerrainMod.LOGGER.error("[Terrain OnnxTerrainGenerator] Bridge probe: unexpected error {}", t.getMessage());
            return false;
        }
    }
    
    /**
     * Get diagnostic information about ONNX terrain generation status.
     * @return Status message for debugging
     */
    public static String getStatusInfo() {
        StringBuilder status = new StringBuilder();
        status.append("ONNX Terrain Status:\n");
        
        try {
            boolean progressiveAvailable = areProgressiveModelsAvailable();
            boolean legacyAvailable = ModelManager.isAvailable();
            
            status.append("- Progressive models available: ").append(progressiveAvailable).append("\n");
            status.append("- Legacy model available: ").append(legacyAvailable).append("\n");
            status.append("- Model available: ").append(progressiveAvailable || legacyAvailable).append("\n");
            status.append("- Configured adapter: ").append(Config.adapter()).append("\n");
            status.append("- Adapter available: ").append(AdapterRegistry.hasAdapter(Config.adapter())).append("\n");
            status.append("- Available adapters: ").append(String.join(", ", AdapterRegistry.getAvailableAdapters()));
            
            if (progressiveAvailable) {
                status.append("\n- Progressive models detected - using 4-stage LOD refinement");
            }
        } catch (Exception e) {
            status.append("- Error getting status: ").append(e.getMessage());
        }
        
        return status.toString();
    }
    
    /**
     * Extract heightmap from a chunk for ONNX processing.
     */
    private int[][] extractHeightmapFromChunk(Chunk chunk) {
        int[][] heightmap = new int[16][16];
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                // Find the highest non-air block
                int highestY = 320; // World top Y in 1.21
                for (int y = 320; y >= -64; y--) { // World height range in 1.21
                    BlockPos pos = new BlockPos(x, y, z);
                    if (!chunk.getBlockState(pos).isAir()) {
                        highestY = y;
                        break;
                    }
                }
                heightmap[x][z] = highestY;
            }
        }
        
        return heightmap;
    }
    
    /**
     * Extract biome IDs from a chunk for ONNX processing.
     */
    private int[][] extractBiomeIdsFromChunk(Chunk chunk) {
        int[][] biomeIds = new int[16][16];
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                // Sample biome at surface level (y=64 as default)
                var biome = chunk.getBiomeForNoiseGen(x, 64, z);
                // Convert biome to simple ID (this could be improved with proper biome registry)
                biomeIds[x][z] = biome.hashCode() % 10; // Simple mapping for now
            }
        }
        
        return biomeIds;
    }
    
    /**
     * Calculate base Y coordinate for terrain generation.
     */
    private int calculateBaseY(int[][] heightmap) {
        int sum = 0;
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                sum += heightmap[x][z];
            }
        }
        return sum / 256; // Average height
    }
    
    /**
     * Apply generated terrain back to the chunk.
     */
    private void applyTerrainToChunk(Chunk chunk, int[][][] terrain, int baseY) {
        // Apply the 16x16x16 terrain to the chunk starting at baseY
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 16; y++) {
                    int worldY = baseY + y;
                    if (worldY >= -64 && worldY < 320) { // World height limits in 1.21
                        int blockType = terrain[x][z][y];
                        
                        // Convert block type ID to actual block state
                        // This is a simplified mapping - could be improved
                        net.minecraft.block.BlockState blockState;
                        if (blockType == 0) {
                            blockState = net.minecraft.block.Blocks.AIR.getDefaultState();
                        } else if (blockType == 1) {
                            blockState = net.minecraft.block.Blocks.STONE.getDefaultState();
                        } else if (blockType == 2) {
                            blockState = net.minecraft.block.Blocks.DIRT.getDefaultState();
                        } else if (blockType == 3) {
                            blockState = net.minecraft.block.Blocks.GRASS_BLOCK.getDefaultState();
                        } else {
                            blockState = net.minecraft.block.Blocks.STONE.getDefaultState(); // Default
                        }
                        
                        BlockPos pos = new BlockPos(x, worldY, z);
                        chunk.setBlockState(pos, blockState, false);
                    }
                }
            }
        }
    }
}
