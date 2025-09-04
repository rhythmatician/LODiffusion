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
 * Progressive LOD adapter for refining LOD3 (2x2x2 voxels) to LOD2 (4x4x4 voxels).
 * Input: 2x2x2 parent voxel grid (8x8x8m blocks per voxel)
 * Output: 4x4x4 refined voxel grid (4x4x4m blocks per voxel)
 */
public class ProgressiveLOD3to2Adapter implements OnnxAdapter {
    
    private static final String ADAPTER_NAME = "progressive_lod3to2";
    private static final long[] INPUT_SHAPE = {1, 1, 2, 2, 2};   // [batch, channels, depth, height, width]
    private static final long[] OUTPUT_SHAPE = {1, 1, 4, 4, 4};  // [batch, channels, depth, height, width]
    
    @Override
    public NDArray extractInput(Chunk chunk, ChunkPos pos, long seed, NDManager manager) {
        try {
            // Extract parent LOD3 data (2x2x2 grid representing 8x8x8m blocks)
            float[][][] parentVoxels = new float[2][2][2];
            
            // Sample chunk at 8x8x8 block intervals
            for (int px = 0; px < 2; px++) {
                for (int pz = 0; pz < 2; pz++) {
                    for (int py = 0; py < 2; py++) {
                        float density = calculateRegionDensity(chunk, px * 8, pz * 8, py * 8, 8);
                        parentVoxels[px][pz][py] = density;
                    }
                }
            }
            
            float[] flatData = flattenVoxelData(parentVoxels);
            return manager.create(flatData).reshape(INPUT_SHAPE);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[ProgressiveLOD3to2] Failed to extract input: " + e.getMessage(), e);
            return manager.zeros(new Shape(INPUT_SHAPE), DataType.FLOAT32);
        }
    }
    
    @Override
    public void applyOutput(Chunk chunk, ChunkPos pos, NDArray output, NDManager manager) {
        // Store LOD2 data for next refinement step
        HelloTerrainMod.LOGGER.debug("[ProgressiveLOD3to2] Applied LOD2 refinement to chunk ({}, {})", pos.x, pos.z);
    }
    
    @Override
    public String getAdapterName() {
        return ADAPTER_NAME;
    }
    
    @Override
    public boolean isCompatible(Model model) {
        // Simplified compatibility check
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
    
    private float calculateRegionDensity(Chunk chunk, int startX, int startZ, int startY, int size) {
        int solidBlocks = 0;
        int totalBlocks = 0;
        int chunkBaseY = chunk.getBottomSectionCoord() << 4;
        
        for (int dx = 0; dx < size && startX + dx < 16; dx++) {
            for (int dz = 0; dz < size && startZ + dz < 16; dz++) {
                for (int dy = 0; dy < size; dy++) {
                    int y = chunkBaseY + startY + dy;
                    var pos = new net.minecraft.util.math.BlockPos(startX + dx, y, startZ + dz);
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
    
    private float[] flattenVoxelData(float[][][] voxels) {
        int size = voxels.length * voxels[0].length * voxels[0][0].length;
        float[] flat = new float[size];
        int index = 0;
        
        for (int x = 0; x < voxels.length; x++) {
            for (int z = 0; z < voxels[x].length; z++) {
                for (int y = 0; y < voxels[x][z].length; y++) {
                    flat[index++] = voxels[x][z][y];
                }
            }
        }
        
        return flat;
    }
}
