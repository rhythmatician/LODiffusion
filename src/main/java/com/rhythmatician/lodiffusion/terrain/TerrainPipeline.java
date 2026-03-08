package com.rhythmatician.lodiffusion.terrain;

import static com.rhythmatician.lodiffusion.world.noise.NoiseTap.PerformanceTier.CORE;

import java.io.IOException;
import java.nio.file.Path;

import com.rhythmatician.lodiffusion.cache.FeatureCache;
import com.rhythmatician.lodiffusion.onnx.ModelOrchestrator;
import com.rhythmatician.lodiffusion.onnx.ProgressiveLODPipeline;
import com.rhythmatician.lodiffusion.world.noise.NoiseTap;

import ai.djl.ndarray.NDManager;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.biome.source.BiomeAccess;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.gen.noise.NoiseConfig;

/**
 * High-level terrain pipeline: capture → cache → infer (to target LOD 1-4) → write.
 * LOD0 is left to vanilla Minecraft.
 */
public final class TerrainPipeline implements AutoCloseable {
    // Logger can be added when wiring chunk write paths

    private final FeatureCache cache;
    private final ModelOrchestrator orchestrator;
    private final CarveAdapter carve;
    private final NDManager manager;

    public TerrainPipeline(FeatureCache cache,
                           ModelOrchestrator orchestrator,
                           CarveAdapter carve,
                           NDManager manager) {
        this.cache = cache;
        this.orchestrator = orchestrator;
        this.carve = carve;
        this.manager = manager;
    }

    public static TerrainPipeline createDefault(NDManager manager,
                                                Path initModel, Path initCfg,
                                                Path m12Model, Path m12Cfg,
                                                Path m24Model, Path m24Cfg,
                                                Path m48Model, Path m48Cfg)
            throws IOException, ai.djl.translate.TranslateException {
        FeatureCache fc = new FeatureCache();
        ModelOrchestrator mo = ModelOrchestrator.loadAll(manager, initModel, initCfg, m12Model, m12Cfg,
                m24Model, m24Cfg, m48Model, m48Cfg);
        return new TerrainPipeline(fc, mo, CarveAdapter.NOOP, manager);
    }

    public ProgressiveLODPipeline.GenerationResult generateForChunk(Chunk chunk,
                                                                    NoiseConfig noiseCfg,
                                                                    BiomeAccess biomeAccess,
                                                                    long worldSeed,
                                                                    int targetLod) throws ai.djl.translate.TranslateException {
        ChunkPos pos = chunk.getPos();
        NoiseTap.Cache feat = cache.get(pos);
        if (feat == null) {
            NoiseTap tap = NoiseTap.bind(chunk, noiseCfg, biomeAccess, worldSeed);
            feat = tap.captureAll(NoiseTap.getTierFields(CORE), NoiseTap.getDefaultHeightmaps());
            cache.put(pos, feat);
        }

        ProgressiveLODPipeline.GenerationResult res = orchestrator.run(feat, targetLod);

        // TODO: write res.blockLogits/airMask into chunk (respect target LOD)

        if (targetLod == 0) {
            carve.carve(chunk); // vanilla carve gate at LOD0 only
        }
        return res;
    }

    @Override
    public void close() {
        if (orchestrator != null) orchestrator.close();
        if (manager != null) manager.close();
    }
}
