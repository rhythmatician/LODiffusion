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
 * Progressive LOD adapter for refining LOD1 (8x8x8 voxels) to LOD0 (16x16x16 voxels).
 * Input: 8x8x8 parent voxel grid (2x2x2m blocks per voxel)
 * Output: 16x16x16 refined voxel grid (1x1x1m blocks per voxel)
 * 
 * This is the final refinement step from 2m resolution to block resolution.
 */
public class ProgressiveLOD1to0Adapter implements OnnxAdapter {
    
    private static final String ADAPTER_NAME = "progressive_lod1to0";
    private static final long[] INPUT_SHAPE = {1, 1, 8, 8, 8};    // [batch, channels, depth, height, width]
    private static final long[] OUTPUT_SHAPE = {1, 1, 16, 16, 16}; // [batch, channels, depth, height, width]
    
    // Block encoding for voxel data
    private static final float AIR_VALUE = 0.0f;
    private static final float STONE_VALUE = 1.0f;
    private static final float DIRT_VALUE = 0.5f;
    private static final float GRASS_VALUE = 0.7f;
    
    @Override
    public NDArray extractInput(Chunk chunk, ChunkPos pos, long seed, NDManager manager) {
        try {
            // Extract parent LOD1 data (8x8x8 grid representing 2x2x2m blocks)
            float[][][] parentVoxels = extractParentLOD1Data(chunk);
            
            // Convert to NDArray with proper shape [1, 1, 8, 8, 8]
            float[] flatData = flattenVoxelData(parentVoxels);
            return manager.create(flatData).reshape(INPUT_SHAPE);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[ProgressiveLOD1to0] Failed to extract input: " + e.getMessage(), e);
            // Return zeros as fallback
            return manager.zeros(new Shape(INPUT_SHAPE), DataType.FLOAT32);
        }
    }
    
    @Override
    public void applyOutput(Chunk chunk, ChunkPos pos, NDArray output, NDManager manager) {
        try {
            // Validate output shape
            if (!java.util.Arrays.equals(output.getShape().getShape(), OUTPUT_SHAPE)) {
                HelloTerrainMod.LOGGER.warn("[ProgressiveLOD1to0] Unexpected output shape: " + 
                    java.util.Arrays.toString(output.getShape().getShape()));
                return;
            }
            
            // Convert NDArray to 16x16x16 voxel data
            float[] flatOutput = output.reshape(16 * 16 * 16).toFloatArray();
            float[][][] refinedVoxels = unflattenVoxelData(flatOutput);
            
            // Apply refined voxel data to chunk (block-level precision)
            applyBlockLevelVoxels(chunk, refinedVoxels);
            
            HelloTerrainMod.LOGGER.debug("[ProgressiveLOD1to0] Applied block-level refinement to chunk ({}, {})", pos.x, pos.z);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[ProgressiveLOD1to0] Failed to apply output: " + e.getMessage(), e);
        }
    }
    
    @Override
    public String getAdapterName() {
        return ADAPTER_NAME;
    }
    
    @Override
    public boolean isCompatible(Model model) {
        try {
            if (model == null) return false;
            
            var inputShapes = model.describeInput();
            var outputShapes = model.describeOutput();
            
            // Verify we have exactly one input and one output
            if (inputShapes.size() != 1 || outputShapes.size() != 1) {
                HelloTerrainMod.LOGGER.debug("[ProgressiveLOD1to0] Model has incorrect number of inputs/outputs: {}/{}", 
                    inputShapes.size(), outputShapes.size());
                return false;
            }
            
            // Check input shape matches expected [1, 1, 8, 8, 8]
            var inputDescriptor = inputShapes.values().iterator().next();
            var expectedInputShape = new ai.djl.ndarray.types.Shape(INPUT_SHAPE);
            if (!inputDescriptor.getShape().equals(expectedInputShape)) {
                HelloTerrainMod.LOGGER.debug("[ProgressiveLOD1to0] Input shape mismatch. Expected: {}, Got: {}", 
                    expectedInputShape, inputDescriptor.getShape());
                return false;
            }
            
            // Check output shape matches expected [1, 1, 16, 16, 16]
            var outputDescriptor = outputShapes.values().iterator().next();
            var expectedOutputShape = new ai.djl.ndarray.types.Shape(OUTPUT_SHAPE);
            if (!outputDescriptor.getShape().equals(expectedOutputShape)) {
                HelloTerrainMod.LOGGER.debug("[ProgressiveLOD1to0] Output shape mismatch. Expected: {}, Got: {}", 
                    expectedOutputShape, outputDescriptor.getShape());
                return false;
            }
            
            HelloTerrainMod.LOGGER.debug("[ProgressiveLOD1to0] Model compatibility validated successfully");
            return true;
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.warn("[ProgressiveLOD1to0] Model compatibility check failed: " + e.getMessage());
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
    
    /**
     * Extract parent LOD1 data by sampling 2x2x2 block regions into 8x8x8 voxels.
     */
    private float[][][] extractParentLOD1Data(Chunk chunk) {
        float[][][] parentVoxels = new float[8][8][8];
        int chunkBaseY = chunk.getBottomSectionCoord() << 4;
        
        for (int px = 0; px < 8; px++) {
            for (int pz = 0; pz < 8; pz++) {
                for (int py = 0; py < 8; py++) {
                    // Each parent voxel represents a 2x2x2 block region
                    float blockSum = 0.0f;
                    int blockCount = 0;
                    
                    for (int dx = 0; dx < 2; dx++) {
                        for (int dz = 0; dz < 2; dz++) {
                            for (int dy = 0; dy < 2; dy++) {
                                int worldX = px * 2 + dx;
                                int worldZ = pz * 2 + dz;
                                int worldY = chunkBaseY + py * 2 + dy;
                                
                                if (worldX < 16 && worldZ < 16) {
                                    BlockPos pos = new BlockPos(worldX, worldY, worldZ);
                                    BlockState state = chunk.getBlockState(pos);
                                    blockSum += encodeBlockState(state);
                                    blockCount++;
                                }
                            }
                        }
                    }
                    
                    // Average the 2x2x2 region to get parent voxel value
                    parentVoxels[px][pz][py] = blockCount > 0 ? blockSum / blockCount : 0.0f;
                }
            }
        }
        
        return parentVoxels;
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
    
    /**
     * Apply 16x16x16 refined voxel data directly to chunk blocks.
     */
    private void applyBlockLevelVoxels(Chunk chunk, float[][][] voxels16x16x16) {
        int chunkBaseY = chunk.getBottomSectionCoord() << 4;
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                for (int y = 0; y < 16; y++) {
                    int worldY = chunkBaseY + y;
                    
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
        
        // Update heightmaps after block modifications
        Heightmap.populateHeightmaps(chunk, 
            java.util.EnumSet.of(Heightmap.Type.WORLD_SURFACE));
    }
}
