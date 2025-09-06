package com.rhythmatician.lodiffusion.onnx;

import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.logging.Logger;

import com.rhythmatician.lodiffusion.world.noise.NoiseTap;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

/**
 * Progressive LOD generation pipeline that chains five ONNX models:
 * Init (Noise→LOD4) → LOD4→LOD3 (1³→2³) → LOD3→LOD2 (2³→4³) → LOD2→LOD1 (4³→8³) → LOD1→LOD0 (8³→16³)
 *
 * Exact inputs are passed without upsampling; each model handles resizing internally with static ops.
 */
public class ProgressiveLODPipeline implements AutoCloseable {

    private static final Logger LOGGER = Logger.getLogger(ProgressiveLODPipeline.class.getName());

    // Stages
    public static final int STAGE_INIT = 0;        // Noise → LOD4 (1³)
    public static final int STAGE_LOD4_TO_LOD3 = 1; // 1³ → 2³
    public static final int STAGE_LOD3_TO_LOD2 = 2; // 2³ → 4³
    public static final int STAGE_LOD2_TO_LOD1 = 3; // 4³ → 8³
    public static final int STAGE_LOD1_TO_LOD0 = 4; // 8³ → 16³

    public record ModelSession(
        ZooModel<NDList, NDList> model,
        ModelConfig config,
        int inputResolution,
        int outputResolution,
        String name
    ) implements AutoCloseable {
        @Override public void close() { if (model != null) model.close(); }
    }

    public record GenerationResult(
        float[][][][][] blockLogits,  // [1][N][D][D][D]
        float[][][][][] airMask,      // [1][1][D][D][D]
        long totalTimeMs,
        long[] modelTimesMs,          // [5]
        int[] resolutions             // [1,2,4,8,16]
    ) {}

    private final NDManager manager;
    private final ModelSession[] models;

    public ProgressiveLODPipeline(NDManager manager, ModelSession[] models) {
        if (models.length != 5) {
            throw new IllegalArgumentException("Progressive LOD pipeline requires exactly 5 models");
        }
        this.manager = manager;
        this.models = models;
        validatePipeline();
    }

    public GenerationResult generate(NoiseTap.Cache cache) throws TranslateException {
        return generateToLOD(cache, 0);
    }

    public GenerationResult generateToLOD(NoiseTap.Cache cache, int targetLod) throws TranslateException {
        if (targetLod < 0 || targetLod > 4) {
            throw new IllegalArgumentException("Target LOD must be 0-4");
        }

        long start = System.currentTimeMillis();
        long[] times = new long[5];
        int[] resolutions = {1, 2, 4, 8, 16};

        NDArray parent = null; // x_parent_prev propagates; null for init
        float[][][][][] lastLogits = null;
        float[][][][][] lastAir = null;

        for (int stage = 0; stage < 5; stage++) {
            if (stage < 4 && (4 - stage) < targetLod) {
                // Skip further refinement if we already reached target LOD
                break;
            }

            long t0 = System.currentTimeMillis();

            ModelSession s = models[stage];
            ProgressiveLODInputBuilder ib = new ProgressiveLODInputBuilder(manager, s.config());
            Map<String, NDArray> inputMap = ib.buildInputs(cache, parent, stage);
            NDList inputs = toNamedNDList(inputMap, s.config());
            try (var predictor = s.model().newPredictor(new NoopTranslator())) {
                NDList out = predictor.predict(inputs);
                
                // Pick outputs by shape (fallback approach)
                // block_logits: [1, N_blocks, D, D, D] - the large channel count 
                // air_mask: [1, 1, D, D, D] - the single channel
                NDArray blockLogits = null;
                NDArray airMask = null;
                
                for (NDArray tensor : out) {
                    long[] shape = tensor.getShape().getShape();
                    if (shape.length == 5 && shape[0] == 1) {
                        long channels = shape[1];
                        if (channels > 100) {  // block_logits has ~1104 channels
                            blockLogits = tensor;
                        } else if (channels == 1) {  // air_mask has 1 channel
                            airMask = tensor;
                        }
                    }
                }
                
                if (blockLogits == null) {
                    throw new IllegalStateException("Could not find block_logits output tensor");
                }
                if (airMask == null) {
                    throw new IllegalStateException("Could not find air_mask output tensor");
                }

                // Debug: Print actual output shapes
                System.out.println("=== DEBUG: ONNX OUTPUT SELECTION ===");
                System.out.println("Model: " + s.config().modelName());
                System.out.println("Selected block_logits shape: " + java.util.Arrays.toString(blockLogits.getShape().getShape()));
                System.out.println("Selected air_mask shape: " + java.util.Arrays.toString(airMask.getShape().getShape()));

                lastLogits = extractTensor5D(blockLogits);
                lastAir = extractTensor5D(airMask);

                parent = blockLogits; // propagate as x_parent_prev to the next stage
            }

            times[stage] = System.currentTimeMillis() - t0;

            if (stage == 4 - targetLod) {
                break; // reached target resolution
            }
        }

        long total = System.currentTimeMillis() - start;
        return new GenerationResult(lastLogits, lastAir, total, times, resolutions);
    }

    private NDList toNamedNDList(Map<String, NDArray> inputs, ModelConfig cfg) {
        // Respect required inputs order by iterating over known keys; names ensure mapping
        List<NDArray> arrs = new ArrayList<>();
        for (String key : cfg.inputs().keySet()) {
            NDArray a = inputs.get(key);
            if (a == null) throw new IllegalArgumentException("Missing required input: " + key);
            a.setName(key);
            arrs.add(a);
        }
        // Append optional inputs if present
        for (String key : cfg.optionalInputs().keySet()) {
            NDArray a = inputs.get(key);
            if (a != null) {
                a.setName(key);
                arrs.add(a);
            }
        }
        return new NDList(arrs);
    }

    private float[][][][][] deriveAirFromLogits(float[][][][][] logits) {
        // Fallback: use channel 0 as air probability
        int b = logits.length;
        int c = logits[0].length;
        int d = logits[0][0].length;
        int h = logits[0][0][0].length;
        int w = logits[0][0][0][0].length;
        float[][][][][] air = new float[b][1][d][h][w];
        for (int di = 0; di < d; di++)
            for (int hi = 0; hi < h; hi++)
                for (int wi = 0; wi < w; wi++)
                    air[0][0][di][hi][wi] = logits[0][0][di][hi][wi];
        return air;
    }

    private float[][][][][] extractTensor5D(NDArray t) {
        long[] s = t.getShape().getShape();
        if (s.length != 5) throw new IllegalArgumentException("Expected 5D tensor, got " + s.length + "D");
        int b = (int) s[0], c = (int) s[1], d = (int) s[2], h = (int) s[3], w = (int) s[4];
        float[] flat = t.toFloatArray();
        float[][][][][] out = new float[b][c][d][h][w];
        int idx = 0;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int di = 0; di < d; di++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            out[bi][ci][di][hi][wi] = flat[idx++];
        return out;
    }

    private void validatePipeline() {
        int prevOut = 0; // Previous model's output resolution
        for (int i = 0; i < models.length; i++) {
            int parent = models[i].config().getParentResolution();
            int out = models[i].config().getOutputResolution();
            if (i == 0) {
                // Init model: parent is placeholder, output should be 1³
                if (out != 1) {
                    throw new IllegalStateException("Stage 0 (init) must output 1³, got: " + out);
                }
            } else {
                // Progressive models: parent should match previous output, output should be 2x parent
                if (parent != prevOut) {
                    throw new IllegalStateException("Stage " + i + " parent mismatch: expected " + prevOut + ", got " + parent);
                }
                if (out != parent * 2) {
                    throw new IllegalStateException("Stage " + i + " output mismatch: expected " + (parent * 2) + ", got " + out);
                }
            }
            prevOut = out; // Update for next iteration
        }
        LOGGER.info("Validated 5-model progressive LOD pipeline (1→2→4→8→16)");
    }

    @Override
    public void close() {
        for (ModelSession s : models) if (s != null) s.close();
        if (manager != null) manager.close();
    }

    public static class Builder {
        private final NDManager manager;
        private final ModelSession[] models = new ModelSession[5];

        public Builder(NDManager manager) { this.manager = manager; }

        public Builder setModel(int stage, Path modelPath, ModelConfig config, String name) {
            if (stage < 0 || stage >= 5) throw new IllegalArgumentException("Stage must be 0-4");
            Criteria<NDList, NDList> c = Criteria.builder()
                .setTypes(NDList.class, NDList.class)
                .optModelPath(modelPath)
                .optTranslator(new NoopTranslator())
                .build();
            try {
                ZooModel<NDList, NDList> model = c.loadModel();
                models[stage] = new ModelSession(
                    model,
                    config,
                    config.getParentResolution(),
                    config.getOutputResolution(),
                    name
                );
            } catch (Exception e) {
                throw new RuntimeException("Failed to load model for stage " + stage + " from " + modelPath, e);
            }
            return this;
        }

        public ProgressiveLODPipeline build() {
            for (int i = 0; i < 5; i++) if (models[i] == null) throw new IllegalStateException("Missing model stage " + i);
            return new ProgressiveLODPipeline(manager, models);
        }
    }
}
