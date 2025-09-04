package com.rhythmatician.lodiffusion.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.HelloTerrainMod;
import com.rhythmatician.lodiffusion.terrain.TerrainGenerator;
import com.rhythmatician.lodiffusion.terrain.VanillaLikeTerrainGenerator;
import com.rhythmatician.lodiffusion.terrain.infer.ModelManager;

import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.StructureWorldAccess;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.chunk.ChunkGenerator;

@Mixin(ChunkGenerator.class)
public class ChunkGeneratorMixin {
    
    private static final TerrainGenerator FALLBACK_GENERATOR = new VanillaLikeTerrainGenerator();
    
    /**
     * Inject into the surface generation step to apply our terrain generation.
     * Uses ONNX if available and enabled, falls back to vanilla-like generation otherwise.
     */
    @Inject(method = "generateFeatures", at = @At("TAIL"))
    private void onGenerateFeatures(StructureWorldAccess world, Chunk chunk, StructureAccessor structureAccessor, CallbackInfo ci) {
        try {
            ChunkPos pos = chunk.getPos();
            
            // Check if ONNX terrain generation is enabled
            if (Config.useOnnxTerrain() && ModelManager.isAvailable()) {
                HelloTerrainMod.LOGGER.info("[LODiffusion] Generating chunk at ({}, {}) with ONNX terrain", pos.x, pos.z);
                // TODO: Use OnnxTerrainGenerator when Phase 4 is implemented
                // For now, fall back to vanilla-like
                FALLBACK_GENERATOR.generateChunk(pos, chunk, world.getSeed());
                HelloTerrainMod.LOGGER.debug("[LODiffusion] ONNX terrain generation placeholder - using fallback");
            } else {
                // Use fallback terrain generation
                HelloTerrainMod.LOGGER.info("[LODiffusion] Generating chunk at ({}, {}) with fallback terrain", pos.x, pos.z);
                FALLBACK_GENERATOR.generateChunk(pos, chunk, world.getSeed());
                HelloTerrainMod.LOGGER.debug("[LODiffusion] Applied fallback terrain generation to chunk ({}, {})", pos.x, pos.z);
            }
            
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.error("[LODiffusion] Error generating terrain for chunk: " + e.getMessage(), e);
            // Emergency fallback - do nothing to avoid corrupting world generation
        }
    }
}
