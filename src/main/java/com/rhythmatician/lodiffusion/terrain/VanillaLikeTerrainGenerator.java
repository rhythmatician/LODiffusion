package com.rhythmatician.lodiffusion.terrain;

import com.rhythmatician.lodiffusion.DiffusionChunkGenerator;

import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.chunk.Chunk;

/**
 * Fallback terrain generator that uses the existing diffusion logic.
 * Acts as a bridge to maintain existing behavior when ONNX is disabled.
 */
public final class VanillaLikeTerrainGenerator implements TerrainGenerator {
    private final DiffusionChunkGenerator diffusionGenerator = new DiffusionChunkGenerator();

    @Override
    public void generateChunk(ChunkPos pos, Chunk chunk, long seed) {
        // Extract heightmap from the chunk (basic implementation)
        int[][] heightmap = extractHeightmap(chunk);
        
        // Extract biome information (simplified)
        String[] biomes = extractBiomes(chunk, pos);
        
        // Use existing diffusion logic
        diffusionGenerator.buildSurface(pos.x, pos.z, heightmap, biomes);
        
        // Apply heightmap back to chunk (simplified)
        applyHeightmap(chunk, heightmap);
    }

    private int[][] extractHeightmap(Chunk chunk) {
        int[][] heightmap = new int[16][16];
        var surfaceHeightmap = chunk.getHeightmap(net.minecraft.world.Heightmap.Type.WORLD_SURFACE);
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = surfaceHeightmap.get(x, z);
            }
        }
        
        return heightmap;
    }

    private String[] extractBiomes(Chunk chunk, ChunkPos pos) {
        String[] biomes = new String[256]; // 16x16 biome array
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                // Use a reasonable Y level for biome sampling
                var biome = chunk.getBiomeForNoiseGen(x, 64, z);
                biomes[x * 16 + z] = biome.toString();
            }
        }
        
        return biomes;
    }

    private void applyHeightmap(Chunk chunk, int[][] heightmap) {
        // For now, apply subtle height modifications (limited to avoid breaking terrain)
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int targetHeight = heightmap[x][z];
                int currentHeight = chunk.getHeightmap(net.minecraft.world.Heightmap.Type.WORLD_SURFACE).get(x, z);
                
                // Apply subtle height modifications (within ±2 blocks for safety)
                int heightDiff = Math.max(-2, Math.min(2, targetHeight - currentHeight));
                
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
                    var belowPos = new net.minecraft.util.math.BlockPos(x, y - 1, z);
                    var currentPos = new net.minecraft.util.math.BlockPos(x, y, z);
                    var stateBelow = chunk.getBlockState(belowPos);
                    chunk.setBlockState(currentPos, stateBelow, false);
                }
            }
        } else if (heightDiff < 0) {
            // Remove blocks (lower terrain)
            for (int i = 0; i < -heightDiff; i++) {
                int y = currentHeight - 1 - i;
                if (y >= chunk.getBottomSectionCoord() << 4) {
                    var currentPos = new net.minecraft.util.math.BlockPos(x, y, z);
                    chunk.setBlockState(currentPos, net.minecraft.block.Blocks.AIR.getDefaultState(), false);
                }
            }
        }
        
        // Update heightmaps after modification
        int finalY = currentHeight + heightDiff;
        var finalPos = new net.minecraft.util.math.BlockPos(x, finalY, z);
        var finalState = chunk.getBlockState(finalPos);
        chunk.getHeightmap(net.minecraft.world.Heightmap.Type.WORLD_SURFACE).trackUpdate(x, finalY, z, finalState);
    }
}
