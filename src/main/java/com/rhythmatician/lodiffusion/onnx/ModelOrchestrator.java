package com.rhythmatician.lodiffusion.onnx;

import java.io.IOException;
import java.nio.file.Path;
import java.util.logging.Logger;

import com.rhythmatician.lodiffusion.world.noise.NoiseTap;

import ai.djl.ndarray.NDManager;
import ai.djl.translate.TranslateException;

/**
 * Loads and runs the five models in sequence, returning the final NDArray voxels.
 */
public final class ModelOrchestrator implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(ModelOrchestrator.class.getName());

    private final NDManager manager;
    private final ProgressiveLODPipeline pipeline;

    private ModelOrchestrator(NDManager manager, ProgressiveLODPipeline pipeline) {
        this.manager = manager;
        this.pipeline = pipeline;
    }

    public static ModelOrchestrator loadAll(NDManager manager,
                                            Path initModel, Path initConfig,
                                            Path m12Model, Path m12Config,
                                            Path m24Model, Path m24Config,
                                            Path m48Model, Path m48Config,
                                            Path m816Model, Path m816Config)
            throws IOException, TranslateException {

        ProgressiveLODPipeline.Builder b = new ProgressiveLODPipeline.Builder(manager);
        b.setModel(ProgressiveLODPipeline.STAGE_INIT, initModel, ConfigLoader.load(initConfig), "Init->LOD4");
        b.setModel(ProgressiveLODPipeline.STAGE_LOD4_TO_LOD3, m12Model, ConfigLoader.load(m12Config), "LOD4->LOD3");
        b.setModel(ProgressiveLODPipeline.STAGE_LOD3_TO_LOD2, m24Model, ConfigLoader.load(m24Config), "LOD3->LOD2");
        b.setModel(ProgressiveLODPipeline.STAGE_LOD2_TO_LOD1, m48Model, ConfigLoader.load(m48Config), "LOD2->LOD1");
        b.setModel(ProgressiveLODPipeline.STAGE_LOD1_TO_LOD0, m816Model, ConfigLoader.load(m816Config), "LOD1->LOD0");

        return new ModelOrchestrator(manager, b.build());
    }

    public ProgressiveLODPipeline.GenerationResult run(NoiseTap.Cache cache, int targetLod)
            throws TranslateException {
        return pipeline.generateToLOD(cache, targetLod);
    }

    @Override
    public void close() {
        if (pipeline != null) pipeline.close();
        if (manager != null) manager.close();
    }
}
