package com.rhythmatician.lodiffusion.terrain.adapter;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.Chunk;

/**
 * Base interface for adapters that bridge between Minecraft chunks and ONNX model I/O.
 * Each adapter handles a specific data format (heightmap8x8, voxel8x8x8, etc.)
 * and encapsulates all encoding/decoding logic.
 */
public interface OnnxAdapter {
    
    /**
     * Extract model input data from a Minecraft chunk.
     * @param chunk Source chunk
     * @param pos Chunk position
     * @param seed World seed for deterministic features
     * @param manager NDArray manager for tensor allocation
     * @return Model input tensor ready for inference
     */
    NDArray extractInput(Chunk chunk, ChunkPos pos, long seed, NDManager manager);
    
    /**
     * Apply model output back to a Minecraft chunk.
     * @param chunk Target chunk to modify
     * @param pos Chunk position
     * @param output Model output tensor
     * @param manager NDArray manager for tensor operations
     */
    void applyOutput(Chunk chunk, ChunkPos pos, NDArray output, NDManager manager);
    
    /**
     * Get the adapter's unique identifier (matches config.json adapter field).
     * @return Adapter name (e.g., "heightmap8x8", "voxel8x8x8")
     */
    String getAdapterName();
    
    /**
     * Validate that this adapter is compatible with the given model.
     * Checks input/output shapes and data types.
     * @param model ONNX model to validate against
     * @return true if compatible, false otherwise
     */
    boolean isCompatible(Model model);
    
    /**
     * Get expected input shape for this adapter.
     * @return Array representing expected input dimensions
     */
    long[] getExpectedInputShape();
    
    /**
     * Get expected output shape for this adapter.
     * @return Array representing expected output dimensions
     */
    long[] getExpectedOutputShape();
}
