package com.rhythmatician.lodiffusion.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.HelloTerrainMod;
import com.rhythmatician.lodiffusion.terrain.OnnxTerrainGenerator;
import com.rhythmatician.lodiffusion.terrain.TerrainGenerator;
import com.rhythmatician.lodiffusion.terrain.VanillaLikeTerrainGenerator;

import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.StructureWorldAccess;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.chunk.ChunkGenerator;

@Mixin(ChunkGenerator.class)
public class ChunkGeneratorMixin {
    
    private static final TerrainGenerator ONNX_GENERATOR = new OnnxTerrainGenerator();
    private static final TerrainGenerator FALLBACK_GENERATOR = new VanillaLikeTerrainGenerator();
    
    /**
     * Inject into the surface generation step to apply our terrain generation.
     * Uses ONNX if available and enabled, fails hard otherwise.
     */
    @Inject(method = "generateFeatures", at = @At("TAIL"))
    private void onGenerateFeatures(StructureWorldAccess world, Chunk chunk, StructureAccessor structureAccessor, CallbackInfo ci) {
        ChunkPos pos = chunk.getPos();
        
        // Check if ONNX terrain generation is enabled
        if (Config.useOnnxTerrain() && OnnxTerrainGenerator.isReady()) {
            HelloTerrainMod.LOGGER.info("[LODiffusion] Generating chunk at ({}, {}) with ONNX terrain", pos.x, pos.z);
            ONNX_GENERATOR.generateChunk(pos, chunk, world.getSeed());
            HelloTerrainMod.LOGGER.debug("[LODiffusion] Applied ONNX terrain generation to chunk ({}, {})", pos.x, pos.z);
        } else {
            // Temporary fallback during testing - will be removed once ONNX is stable
            String reason = !Config.useOnnxTerrain() ? "ONNX terrain disabled in config" : "ONNX generator not ready";
            HelloTerrainMod.LOGGER.warn("[LODiffusion] {} for chunk ({}, {}) - panic!", reason, pos.x, pos.z);
            // FALLBACK_GENERATOR.generateChunk(pos, chunk, world.getSeed());
            throw new IllegalStateException("Terrain generation failed");
        }
    }
}
