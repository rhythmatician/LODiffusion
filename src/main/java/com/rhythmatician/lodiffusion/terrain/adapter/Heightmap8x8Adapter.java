package com.rhythmatician.lodiffusion.terrain.adapter;

import com.rhythmatician.lodiffusion.HelloTerrainMod;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.Heightmap;
import net.minecraft.world.chunk.Chunk;

/**
 * Adapter for heightmap-based terrain generation.
 * Input: 8x8 heightmap (downsampled from 16x16)
 * Output: 16x16 heightmap (full chunk resolution)
 * Data format: float32, normalized to [-1, 1] range
 */
public class Heightmap8x8Adapter implements OnnxAdapter {
    
    private static final String ADAPTER_NAME = "heightmap8x8";
    private static final long[] INPUT_SHAPE = {1, 1, 8, 8};   // [batch, channels, height, width]
    private static final long[] OUTPUT_SHAPE = {1, 1, 16, 16}; // [batch, channels, height, width]
    
    // Normalization constants for heightmap values
    private static final float HEIGHT_MEAN = 64.0f;    // Typical world height
    private static final float HEIGHT_SCALE = 32.0f;   // Normalization scale
    
    @Override
    public NDArray extractInput(Chunk chunk, ChunkPos pos, long seed, NDManager manager) {
        try {
            // Extract 16x16 heightmap from chunk
            int[][] fullHeightmap = extractFullHeightmap(chunk);
            
            // Downsample to 8x8 for model input
            float[][] input8x8 = downsampleHeightmap(fullHeightmap);
            
            // Normalize to [-1, 1] range
            normalizeHeightmap(input8x8);
            
            // Convert to NDArray with proper shape [1, 1, 8, 8]
            return manager.create(input8x8).reshape(INPUT_SHAPE);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[Heightmap8x8Adapter] Failed to extract input: " + e.getMessage(), e);
            // Return zeros as fallback
            return manager.zeros(new Shape(INPUT_SHAPE), DataType.FLOAT32);
        }
    }
    
    @Override
    public void applyOutput(Chunk chunk, ChunkPos pos, NDArray output, NDManager manager) {
        try {
            // Validate output shape
            if (!java.util.Arrays.equals(output.getShape().getShape(), OUTPUT_SHAPE)) {
                HelloTerrainMod.LOGGER.warn("[Heightmap8x8Adapter] Unexpected output shape: " + 
                    java.util.Arrays.toString(output.getShape().getShape()));
                return;
            }
            
            // Convert NDArray to 16x16 heightmap
            float[] flatOutput = output.reshape(16 * 16).toFloatArray();
            float[][] output16x16 = new float[16][16];
            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    output16x16[i][j] = flatOutput[i * 16 + j];
                }
            }
            
            // Denormalize from [-1, 1] to actual heights
            denormalizeHeightmap(output16x16);
            
            // Apply heightmap to chunk with safety bounds
            applyHeightmapToChunk(chunk, output16x16);
            
            HelloTerrainMod.LOGGER.debug("[Heightmap8x8Adapter] Applied heightmap to chunk ({}, {})", pos.x, pos.z);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[Heightmap8x8Adapter] Failed to apply output: " + e.getMessage(), e);
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
            HelloTerrainMod.LOGGER.warn("[Heightmap8x8Adapter] Model compatibility check failed: " + e.getMessage());
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
    
    private int[][] extractFullHeightmap(Chunk chunk) {
        int[][] heightmap = new int[16][16];
        Heightmap surfaceHeightmap = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE);
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = surfaceHeightmap.get(x, z);
            }
        }
        
        return heightmap;
    }
    
    private float[][] downsampleHeightmap(int[][] full16x16) {
        float[][] downsampled = new float[8][8];
        
        // Average 2x2 blocks to get 8x8 from 16x16
        for (int x = 0; x < 8; x++) {
            for (int z = 0; z < 8; z++) {
                int sum = 0;
                sum += full16x16[x * 2][z * 2];
                sum += full16x16[x * 2 + 1][z * 2];
                sum += full16x16[x * 2][z * 2 + 1];
                sum += full16x16[x * 2 + 1][z * 2 + 1];
                downsampled[x][z] = sum / 4.0f;
            }
        }
        
        return downsampled;
    }
    
    private void normalizeHeightmap(float[][] heightmap) {
        for (int x = 0; x < heightmap.length; x++) {
            for (int z = 0; z < heightmap[x].length; z++) {
                // Normalize: (height - mean) / scale
                heightmap[x][z] = (heightmap[x][z] - HEIGHT_MEAN) / HEIGHT_SCALE;
                // Clamp to [-1, 1]
                heightmap[x][z] = Math.max(-1.0f, Math.min(1.0f, heightmap[x][z]));
            }
        }
    }
    
    private void denormalizeHeightmap(float[][] heightmap) {
        for (int x = 0; x < heightmap.length; x++) {
            for (int z = 0; z < heightmap[x].length; z++) {
                // Denormalize: height * scale + mean
                heightmap[x][z] = heightmap[x][z] * HEIGHT_SCALE + HEIGHT_MEAN;
                // Clamp to reasonable world bounds
                heightmap[x][z] = Math.max(-64.0f, Math.min(320.0f, heightmap[x][z]));
            }
        }
    }
    
    private void applyHeightmapToChunk(Chunk chunk, float[][] heightmap16x16) {
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int targetHeight = Math.round(heightmap16x16[x][z]);
                int currentHeight = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE).get(x, z);
                
                // Apply conservative height modifications (±4 blocks max for safety)
                int heightDiff = Math.max(-4, Math.min(4, targetHeight - currentHeight));
                
                if (heightDiff != 0) {
                    modifyTerrainHeight(chunk, x, z, currentHeight, heightDiff);
                }
            }
        }
    }
    
    private void modifyTerrainHeight(Chunk chunk, int x, int z, int currentHeight, int heightDiff) {
        if (heightDiff > 0) {
            // Add blocks (raise terrain)
            for (int i = 0; i < heightDiff; i++) {
                int y = currentHeight + i;
                if (y < chunk.getTopSectionCoord() << 4) {
                    BlockPos belowPos = new BlockPos(x, y - 1, z);
                    BlockPos currentPos = new BlockPos(x, y, z);
                    var stateBelow = chunk.getBlockState(belowPos);
                    chunk.setBlockState(currentPos, stateBelow, false);
                }
            }
        } else if (heightDiff < 0) {
            // Remove blocks (lower terrain)
            for (int i = 0; i < -heightDiff; i++) {
                int y = currentHeight - 1 - i;
                if (y >= chunk.getBottomSectionCoord() << 4) {
                    BlockPos currentPos = new BlockPos(x, y, z);
                    chunk.setBlockState(currentPos, net.minecraft.block.Blocks.AIR.getDefaultState(), false);
                }
            }
        }
        
        // Update heightmaps after modification
        int finalY = currentHeight + heightDiff;
        BlockPos finalPos = new BlockPos(x, finalY, z);
        var finalState = chunk.getBlockState(finalPos);
        chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE).trackUpdate(x, finalY, z, finalState);
    }
}
