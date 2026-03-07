package com.rhythmatician.lodiffusion.mixin;

import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

import net.minecraft.world.StructureWorldAccess;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.StructureAccessor;
import net.minecraft.world.gen.chunk.ChunkGenerator;

@Mixin(ChunkGenerator.class)
public class ChunkGeneratorMixin {
    
    // DISABLED: The old approach overwrote vanilla chunk generation with ONNX output.
    // LODiffusion now generates LODs proactively via LodGenerationService + VoxySectionWriter
    // and pushes them to Voxy for distant rendering.  Vanilla world gen runs normally.
    
    /**
     * No-op injection point — kept so the mixin class remains valid.
     * Logs the first few chunk generations for debugging connectivity.
     */
    @Inject(method = "generateFeatures", at = @At("TAIL"))
    private void onGenerateFeatures(StructureWorldAccess world, Chunk chunk, StructureAccessor structureAccessor, CallbackInfo ci) {
        // Vanilla world generation proceeds normally — no ONNX override.
        // LOD generation for distant areas is handled by LodGenerationService.
    }
}
