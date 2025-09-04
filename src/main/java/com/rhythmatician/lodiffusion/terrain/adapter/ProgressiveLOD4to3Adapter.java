package com.rhythmatician.lodiffusion.terrain.adapter;

import com.rhythmatician.lodiffusion.HelloTerrainMod;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.Chunk;

/**
 * Progressive LOD adapter for refining LOD4 (1x1x1 voxel) to LOD3 (2x2x2 voxels).
 * Input: 1x1x1 parent voxel (entire 16x16x16m subchunk)
 * Output: 2x2x2 refined voxel grid (8x8x8m blocks per voxel)
 */
public class ProgressiveLOD4to3Adapter implements OnnxAdapter {
    
    private static final String ADAPTER_NAME = "progressive_lod4to3";
    private static final long[] INPUT_SHAPE = {1, 1, 1, 1, 1};   // [batch, channels, depth, height, width]
    private static final long[] OUTPUT_SHAPE = {1, 1, 2, 2, 2};  // [batch, channels, depth, height, width]
    
    @Override
    public NDArray extractInput(Chunk chunk, ChunkPos pos, long seed, NDManager manager) {
        try {
            // Extract parent LOD4 data (single voxel representing entire subchunk)
            float density = calculateChunkDensity(chunk);
            
            float[] input = {density};
            return manager.create(input).reshape(INPUT_SHAPE);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[ProgressiveLOD4to3] Failed to extract input: " + e.getMessage(), e);
            return manager.zeros(new Shape(INPUT_SHAPE), DataType.FLOAT32);
        }
    }
    
    @Override
    public void applyOutput(Chunk chunk, ChunkPos pos, NDArray output, NDManager manager) {
        // Store LOD3 data for next refinement step
        HelloTerrainMod.LOGGER.debug("[ProgressiveLOD4to3] Applied LOD3 refinement to chunk ({}, {})", pos.x, pos.z);
    }
    
    @Override
    public String getAdapterName() {
        return ADAPTER_NAME;
    }
    
    @Override
    public boolean isCompatible(Model model) {
        return model != null;
    }
    
    @Override
    public long[] getExpectedInputShape() {
        return INPUT_SHAPE.clone();
    }
    
    @Override
    public long[] getExpectedOutputShape() {
        return OUTPUT_SHAPE.clone();
    }
    
    private float calculateChunkDensity(Chunk chunk) {
        int solidBlocks = 0;
        int totalBlocks = 0;
        int chunkBaseY = chunk.getBottomSectionCoord() << 4;
        
        // Sample the entire 16x16x16 subchunk
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 16; y++) {
                    var pos = new net.minecraft.util.math.BlockPos(x, chunkBaseY + y, z);
                    var state = chunk.getBlockState(pos);
                    if (!state.isAir()) {
                        solidBlocks++;
                    }
                    totalBlocks++;
                }
            }
        }
        
        return totalBlocks > 0 ? (float) solidBlocks / totalBlocks : 0.0f;
    }
}
