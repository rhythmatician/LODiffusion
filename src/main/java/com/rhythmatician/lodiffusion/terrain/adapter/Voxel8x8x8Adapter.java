package com.rhythmatician.lodiffusion.terrain.adapter;

import com.rhythmatician.lodiffusion.HelloTerrainMod;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import net.minecraft.block.Block;
import net.minecraft.block.BlockState;
import net.minecraft.block.Blocks;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.Heightmap;
import net.minecraft.world.chunk.Chunk;

/**
 * Adapter for full 3D voxel-based terrain generation.
 * Input: 8x8x8 voxel grid (downsampled from 16x16x16)
 * Output: 16x16x16 voxel grid (full chunk resolution)
 * Data format: float32, block type encoding
 */
public class Voxel8x8x8Adapter implements OnnxAdapter {
    
    private static final String ADAPTER_NAME = "voxel8x8x8";
    private static final long[] INPUT_SHAPE = {1, 1, 8, 8, 8};   // [batch, channels, depth, height, width]
    private static final long[] OUTPUT_SHAPE = {1, 1, 16, 16, 16}; // [batch, channels, depth, height, width]
    
    // Block encoding - simplified mapping
    private static final float AIR_VALUE = 0.0f;
    private static final float STONE_VALUE = 1.0f;
    private static final float DIRT_VALUE = 0.5f;
    private static final float GRASS_VALUE = 0.7f;
    
    // Y-level range for voxel sampling (relative to chunk base)
    private static final int MIN_Y_RELATIVE = 0;   // Bedrock level
    
    @Override
    public NDArray extractInput(Chunk chunk, ChunkPos pos, long seed, NDManager manager) {
        try {
            // Extract 16x16x16 voxel data from chunk
            float[][][] fullVoxels = extractFullVoxelData(chunk);
            
            // Downsample to 8x8x8 for model input
            float[][][] input8x8x8 = downsampleVoxelData(fullVoxels);
            
            // Convert to NDArray with proper shape [1, 1, 8, 8, 8]
            float[] flatData = flattenVoxelData(input8x8x8);
            return manager.create(flatData).reshape(INPUT_SHAPE);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[Voxel8x8x8Adapter] Failed to extract input: " + e.getMessage(), e);
            // Return zeros as fallback
            return manager.zeros(new Shape(INPUT_SHAPE), DataType.FLOAT32);
        }
    }
    
    @Override
    public void applyOutput(Chunk chunk, ChunkPos pos, NDArray output, NDManager manager) {
        try {
            // Validate output shape
            if (!java.util.Arrays.equals(output.getShape().getShape(), OUTPUT_SHAPE)) {
                HelloTerrainMod.LOGGER.warn("[Voxel8x8x8Adapter] Unexpected output shape: " + 
                    java.util.Arrays.toString(output.getShape().getShape()));
                return;
            }
            
            // Convert NDArray to 16x16x16 voxel data
            float[] flatOutput = output.reshape(16 * 16 * 16).toFloatArray();
            float[][][] output16x16x16 = unflattenVoxelData(flatOutput);
            
            // Apply voxel data to chunk with safety checks
            applyVoxelDataToChunk(chunk, output16x16x16);
            
            HelloTerrainMod.LOGGER.debug("[Voxel8x8x8Adapter] Applied voxel data to chunk ({}, {})", pos.x, pos.z);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[Voxel8x8x8Adapter] Failed to apply output: " + e.getMessage(), e);
        }
    }
    
    @Override
    public String getAdapterName() {
        return ADAPTER_NAME;
    }
    
    @Override
    public boolean isCompatible(Model model) {
        try {
            // Check if model has correct input/output shapes
            // This is a simplified check - in practice you'd query model metadata
            return true; // TODO: Implement proper model shape validation
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.warn("[Voxel8x8x8Adapter] Model compatibility check failed: " + e.getMessage());
            return false;
        }
    }
    
    @Override
    public long[] getExpectedInputShape() {
        return INPUT_SHAPE.clone();
    }
    
    @Override
    public long[] getExpectedOutputShape() {
        return OUTPUT_SHAPE.clone();
    }
    
    // Private helper methods
    
    private float[][][] extractFullVoxelData(Chunk chunk) {
        float[][][] voxels = new float[16][16][16];
        
        int chunkBaseY = chunk.getBottomSectionCoord() << 4;
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 16; y++) {
                    int worldY = chunkBaseY + MIN_Y_RELATIVE + y;
                    BlockPos pos = new BlockPos(x, worldY, z);
                    BlockState state = chunk.getBlockState(pos);
                    voxels[x][z][y] = encodeBlockState(state);
                }
            }
        }
        
        return voxels;
    }
    
    private float[][][] downsampleVoxelData(float[][][] full16x16x16) {
        float[][][] downsampled = new float[8][8][8];
        
        // Average 2x2x2 blocks to get 8x8x8 from 16x16x16
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                for (int y = 0; y < 8; y++) {
                    float sum = 0.0f;
                    for (int dx = 0; dx < 2; dx++) {
                        for (int dz = 0; dz < 2; dz++) {
                            for (int dy = 0; dy < 2; dy++) {
                                sum += full16x16x16[x * 2 + dx][z * 2 + dz][y * 2 + dy];
                            }
                        }
                    }
                    downsampled[x][z][y] = sum / 8.0f; // Average of 2x2x2 = 8 blocks
                }
            }
        }
        
        return downsampled;
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
    
    private float[][][] unflattenVoxelData(float[] flatData) {
        float[][][] voxels = new float[16][16][16];
        int index = 0;
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 16; y++) {
                    voxels[x][z][y] = flatData[index++];
                }
            }
        }
        
        return voxels;
    }
    
    private float encodeBlockState(BlockState state) {
        Block block = state.getBlock();
        
        if (block == Blocks.AIR) return AIR_VALUE;
        if (block == Blocks.STONE) return STONE_VALUE;
        if (block == Blocks.DIRT) return DIRT_VALUE;
        if (block == Blocks.GRASS_BLOCK) return GRASS_VALUE;
        
        // Default encoding for unknown blocks
        return STONE_VALUE;
    }
    
    private BlockState decodeBlockState(float value) {
        // Simple threshold-based decoding
        if (value < 0.25f) return Blocks.AIR.getDefaultState();
        if (value < 0.6f) return Blocks.DIRT.getDefaultState();
        if (value < 0.85f) return Blocks.GRASS_BLOCK.getDefaultState();
        return Blocks.STONE.getDefaultState();
    }
    
    private void applyVoxelDataToChunk(Chunk chunk, float[][][] voxels16x16x16) {
        int chunkBaseY = chunk.getBottomSectionCoord() << 4;
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 16; y++) {
                    int worldY = chunkBaseY + MIN_Y_RELATIVE + y;
                    
                    // Skip if outside chunk bounds
                    if (worldY < chunk.getBottomSectionCoord() << 4 || 
                        worldY >= chunk.getTopSectionCoord() << 4) {
                        continue;
                    }
                    
                    BlockPos pos = new BlockPos(x, worldY, z);
                    BlockState currentState = chunk.getBlockState(pos);
                    BlockState newState = decodeBlockState(voxels16x16x16[x][z][y]);
                    
                    // Only modify if different (avoid unnecessary updates)
                    if (!currentState.equals(newState)) {
                        chunk.setBlockState(pos, newState, false);
                    }
                }
            }
        }
        
        // Update heightmaps after voxel modifications
        Heightmap.populateHeightmaps(chunk, 
            java.util.EnumSet.of(Heightmap.Type.WORLD_SURFACE));
    }
}
