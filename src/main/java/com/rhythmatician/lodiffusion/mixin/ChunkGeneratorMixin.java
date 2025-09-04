package com.rhythmatician.lodiffusion.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import com.rhythmatician.lodiffusion.DiffusionChunkGenerator;
import com.rhythmatician.lodiffusion.HelloTerrainMod;

import net.minecraft.block.BlockState;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.Heightmap;
import net.minecraft.world.StructureWorldAccess;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.chunk.ChunkGenerator;

@Mixin(ChunkGenerator.class)
public class ChunkGeneratorMixin {
    
    private static final DiffusionChunkGenerator diffusionGenerator = new DiffusionChunkGenerator();
    
    /**
     * Inject into the surface generation step to apply our diffusion model.
     */
    @Inject(method = "generateFeatures", at = @At("TAIL"))
    private void onGenerateFeatures(StructureWorldAccess world, Chunk chunk, StructureAccessor structureAccessor, CallbackInfo ci) {
        try {
            ChunkPos pos = chunk.getPos();
            HelloTerrainMod.LOGGER.info("[LODiffusion] Enhancing chunk at ({}, {}) with AI diffusion", pos.x, pos.z);
            
            // Extract heightmap from the chunk
            int[][] heightmap = extractHeightmap(chunk);
            
            // Extract biome information  
            String[] biomes = extractBiomes(world, pos);
            
            // Determine LOD level based on distance from spawn
            int lod = calculateLODLevel(pos);
            
            // Apply our diffusion model with LOD awareness
            if (lod >= 0) {
                diffusionGenerator.buildSurfaceWithLOD(pos.x, pos.z, heightmap, biomes, lod);
                HelloTerrainMod.LOGGER.debug("[LODiffusion] Applied LOD {} diffusion to chunk ({}, {})", lod, pos.x, pos.z);
            } else {
                // Fallback to standard diffusion if LOD calculation fails
                diffusionGenerator.buildSurface(pos.x, pos.z, heightmap, biomes);
                HelloTerrainMod.LOGGER.debug("[LODiffusion] Applied standard diffusion to chunk ({}, {})", pos.x, pos.z);
            }
            
            // Apply the enhanced heightmap back to the chunk
            applyHeightmap(chunk, heightmap);
            
            HelloTerrainMod.LOGGER.debug("[LODiffusion] Successfully enhanced chunk ({}, {})", pos.x, pos.z);
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[LODiffusion] Error enhancing chunk: " + e.getMessage());
        }
    }
    
    /**
     * Calculate the appropriate LOD level for this chunk based on distance from spawn.
     */
    private int calculateLODLevel(ChunkPos chunkPos) {
        try {
            // For terrain generation, we'll use a simpler distance-based approach
            // since player detection during world generation is complex and not always reliable
            
            // Use distance from world spawn (0,0) for LOD calculation
            double distanceFromSpawn = Math.sqrt(
                chunkPos.getCenterX() * chunkPos.getCenterX() + 
                chunkPos.getCenterZ() * chunkPos.getCenterZ()
            );
            
            // Distant Horizons-style LOD levels based on distance
            if (distanceFromSpawn < 512) return 0;       // High detail - close to spawn
            else if (distanceFromSpawn < 1024) return 1;  // Medium detail
            else if (distanceFromSpawn < 2048) return 2;  // Low detail
            else return 3;                               // Very low detail - far from spawn
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.warn("[LODiffusion] Failed to calculate LOD level: " + e.getMessage());
            return -1; // Signal to use fallback diffusion
        }
    }
    
    /**
     * Extract heightmap data from a chunk.
     */
    private int[][] extractHeightmap(Chunk chunk) {
        int[][] heightmap = new int[16][16];
        Heightmap surfaceHeightmap = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE);
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                heightmap[x][z] = surfaceHeightmap.get(x, z);
            }
        }
        
        return heightmap;
    }
    
    /**
     * Extract biome information from the world.
     */
    private String[] extractBiomes(StructureWorldAccess world, ChunkPos pos) {
        String[] biomes = new String[256]; // 16x16 biome array
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int worldX = pos.getStartX() + x;
                int worldZ = pos.getStartZ() + z;
                int y = 64; // Use a reasonable Y level for biome sampling
                
                BlockPos blockPos = new BlockPos(worldX, y, worldZ);
                var biome = world.getBiome(blockPos);
                String biomeName = biome.toString();
                biomes[x * 16 + z] = biomeName;
            }
        }
        
        return biomes;
    }
    
    /**
     * Apply the enhanced heightmap back to the chunk.
     */
    private void applyHeightmap(Chunk chunk, int[][] heightmap) {
        // For now, we'll apply subtle height modifications
        // This is a simplified implementation - in practice you'd want more sophisticated terrain modification
        
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                int targetHeight = heightmap[x][z];
                int currentHeight = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE).get(x, z);
                
                // Apply subtle height modifications (within ±2 blocks for safety)
                int heightDiff = Math.max(-2, Math.min(2, targetHeight - currentHeight));
                
                if (heightDiff != 0) {
                    modifyTerrainHeight(chunk, x, z, currentHeight, heightDiff);
                }
            }
        }
    }
    
    /**
     * Modify terrain height at a specific position.
     */
    private void modifyTerrainHeight(Chunk chunk, int x, int z, int currentHeight, int heightDiff) {
        if (heightDiff > 0) {
            // Add blocks (raise terrain)
            for (int i = 0; i < heightDiff; i++) {
                int y = currentHeight + i;
                if (y < chunk.getTopSectionCoord() << 4) { // Use section-based bounds
                    BlockPos belowPos = new BlockPos(x, y - 1, z);
                    BlockPos currentPos = new BlockPos(x, y, z);
                    BlockState stateBelow = chunk.getBlockState(belowPos);
                    chunk.setBlockState(currentPos, stateBelow, false);
                }
            }
        } else if (heightDiff < 0) {
            // Remove blocks (lower terrain)
            for (int i = 0; i < -heightDiff; i++) {
                int y = currentHeight - 1 - i;
                if (y >= chunk.getBottomSectionCoord() << 4) { // Use section-based bounds
                    BlockPos currentPos = new BlockPos(x, y, z);
                    chunk.setBlockState(currentPos, net.minecraft.block.Blocks.AIR.getDefaultState(), false);
                }
            }
        }
        
        // Update heightmaps after modification
        int finalY = currentHeight + heightDiff;
        BlockPos finalPos = new BlockPos(x, finalY, z);
        BlockState finalState = chunk.getBlockState(finalPos);
        chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE).trackUpdate(x, finalY, z, finalState);
    }
}
