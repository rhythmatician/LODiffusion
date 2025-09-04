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
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.Chunk;

/**
 * TerrainGenerator implementation that uses ONNX models for terrain generation.
 * Delegates to the appropriate adapter based on configuration.
 */
public class OnnxTerrainGenerator implements TerrainGenerator {
    
    @Override
    public void generateChunk(ChunkPos pos, Chunk chunk, long seed) {
        try (var totalTimer = PerformanceMonitor.startTiming(PerformanceMonitor.TOTAL_GENERATION_TIME)) {
            PerformanceMonitor.incrementCounter(PerformanceMonitor.CHUNKS_GENERATED);
            
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
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[OnnxTerrainGenerator] Error generating terrain for chunk ({}, {}): {}", 
                pos.x, pos.z, e.getMessage(), e);
            PerformanceMonitor.incrementCounter(PerformanceMonitor.MODEL_ERRORS);
            // Don't rethrow - allow fallback behavior
        }
    }
    
    /**
     * Check if ONNX terrain generation is ready to use.
     * @return true if model and adapter are available, false otherwise
     */
    public static boolean isReady() {
        try {
            String adapterName = Config.adapter();
            return ModelManager.isAvailable() && 
                   AdapterRegistry.hasAdapter(adapterName);
        } catch (Exception e) {
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
            status.append("- Model available: ").append(ModelManager.isAvailable()).append("\n");
            status.append("- Configured adapter: ").append(Config.adapter()).append("\n");
            status.append("- Adapter available: ").append(AdapterRegistry.hasAdapter(Config.adapter())).append("\n");
            status.append("- Available adapters: ").append(String.join(", ", AdapterRegistry.getAvailableAdapters()));
        } catch (Exception e) {
            status.append("- Error getting status: ").append(e.getMessage());
        }
        
        return status.toString();
    }
}
