package com.rhythmatician.lodiffusion.onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.logging.Logger;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.translate.NoopTranslator;
import ai.djl.translate.TranslateException;

/**
 * Chains the four progressive ONNX models exported by VoxelTree's
 * {@code scripts/export_lod.py} from a {@code train.py} multi-LOD checkpoint.
 *
 * <h3>Contract: {@code lodiffusion.v3.progressive}</h3>
 *
 * <p>All four ONNX models are exported with <b>dynamic batch</b> dimensions,
 * meaning the batch axis (dim 0) is symbolic and can accept any N≥1.
 * This class supports both single-sample inference ({@link #generate},
 * {@link #generateStage}) and batched inference ({@link #generateStageBatch})
 * where multiple sections are evaluated in a single ONNX call for higher
 * throughput.
 *
 * <pre>
 * Pipeline stages (N = batch size):
 *   init_to_lod4        inputs: (hp[N,5,16,16], biome[N,16,16], y[N])
 *                       output: block_logits[N,V,1,1,1]
 *
 *   refine_lod4_to_lod3 inputs: (hp, biome, y, parent[N,1,1,1,1])
 *                       output: block_logits[N,V,2,2,2]
 *
 *   refine_lod3_to_lod2 inputs: (hp, biome, y, parent[N,1,2,2,2])
 *                       output: block_logits[N,V,4,4,4]
 *
 *   refine_lod2_to_lod1 inputs: (hp, biome, y, parent[N,1,4,4,4])
 *                       output: block_logits[N,V,8,8,8]
 * </pre>
 *
 * <p>Air is represented as block class 0 in the unified softmax — there is
 * no separate air mask output.  Binary solid-occupancy parents for inter-stage
 * propagation are derived from the block logits via argmax: if the predicted
 * class is 0 (air) the parent value is 0; otherwise 1.
 *
 * <p>The final 8³ output is upsampled 2× via nearest-neighbour repetition to
 * produce 16³ outputs that match the shape expected by {@link com.rhythmatician.lodiffusion.voxy.VoxySectionWriter}.
 */
public final class ProgressiveModelRunner implements AutoCloseable {

    private static final Logger LOGGER =
            Logger.getLogger(ProgressiveModelRunner.class.getName());

    /** ONNX file stems (no extension), in pipeline order. */
    private static final String[] STEMS = {
        "init_to_lod4",
        "refine_lod4_to_lod3",
        "refine_lod3_to_lod2",
        "refine_lod2_to_lod1"
    };

    /** Config sidecar JSON names, parallel to STEMS. */
    private static final String[] CONFIG_NAMES = {
        "init_to_lod4_config.json",
        "refine_lod4_to_lod3_config.json",
        "refine_lod3_to_lod2_config.json",
        "refine_lod2_to_lod1_config.json"
    };

    /**
     * Parent input tensor spatial shapes for each stage.  Stage 0 has no parent.
     * The parent is the binary solid-occupancy derived from block logits
     * argmax (class 0 = air → 0, else → 1) of the previous stage.
     *
     * <p>Only the spatial dimensions (channel, D, D, D) are listed here;
     * the batch dimension is prepended dynamically at runtime (1 for
     * single-sample, N for batched).
     */
    static final long[][] PARENT_INPUT_SHAPES = {
        null,                // stage 0: no parent
        {1, 1, 1, 1},       // stage 1: parent from stage 0  (ch=1, 1³)
        {1, 2, 2, 2},       // stage 2: parent from stage 1  (ch=1, 2³)
        {1, 4, 4, 4},       // stage 3: parent from stage 2  (ch=1, 4³)
    };

    /**
     * Result of a single pipeline stage execution.
     *
     * @param solidParentFlat  binary solid parent for the next stage (stages 0-2);
     *                         {@code null} for the final stage
     * @param blockLogits      native-resolution block logits [1][N][D][D][D] for
     *                         writing to the corresponding Voxy level;
     *                         {@code null} when not needed (e.g. old pipeline)
     * @param finalResult      full inference result (stage 3 only); {@code null}
     *                         for intermediate stages
     * @param elapsedMs        wall time for this stage
     */
    public record StageOutput(float[] solidParentFlat,
                              float[][][][][] blockLogits,
                              InferenceResult finalResult,
                              long elapsedMs) {
        /** True if this is from the final stage (3). */
        public boolean isFinal() { return finalResult != null; }
    }

    private final NDManager manager;
    @SuppressWarnings("unchecked")
    private final ZooModel<NDList, NDList>[] models;
    private final ModelConfig[]              configs;
    private final BlockVocabulary            vocabulary;

    /**
     * Thread-local reusable buffers for inference.  Eliminates ~2.5 MB of
     * allocation per section call (the main GC pressure source).
     */
    private final ThreadLocal<InferenceBuffers> threadBuffers = new ThreadLocal<>();

    @SuppressWarnings("unchecked")
    private ProgressiveModelRunner(NDManager manager,
                                   ZooModel<NDList, NDList>[] models,
                                   ModelConfig[] configs,
                                   BlockVocabulary vocabulary) {
        this.manager    = manager;
        this.models     = models;
        this.configs    = configs;
        this.vocabulary = vocabulary;
    }

    // ------------------------------------------------------------------
    // Factory
    // ------------------------------------------------------------------

    /**
     * Load all 4 progressive ONNX models from {@code modelDir}.
     *
     * <p>Before loading, validates that every required file listed in
     * {@code pipeline_manifest.json}'s {@code required_files} array is
     * present.  This catches incomplete deployments (e.g. forgetting to
     * copy the {@code _config.json} sidecars) with a clear error message
     * listing exactly which files are missing.
     *
     * @param modelDir directory containing the {@code .onnx} files and their
     *                 {@code _config.json} sidecars (produced by export_lod.py)
     * @throws IOException if any model or config cannot be loaded
     */
    @SuppressWarnings({"unchecked", "rawtypes"})
    public static ProgressiveModelRunner loadAll(Path modelDir) throws IOException {

        // ---- Up-front deployment validation ----
        validateRequiredFiles(modelDir);

        ZooModel<NDList, NDList>[] models = new ZooModel[4];
        ModelConfig[] configs = new ModelConfig[4];

        NDManager manager = NDManager.newBaseManager();
        try {
            for (int i = 0; i < 4; i++) {
                Path configPath = modelDir.resolve(CONFIG_NAMES[i]);
                configs[i] = ConfigLoader.load(configPath);

                Criteria<NDList, NDList> cr = Criteria.builder()
                        .setTypes(NDList.class, NDList.class)
                        .optModelPath(modelDir)
                        .optModelName(STEMS[i])
                        .optTranslator(new NoopTranslator())
                        .build();
                try {
                    models[i] = cr.loadModel();
                    LOGGER.info("[ProgressiveModelRunner] Loaded " + STEMS[i]
                            + "  (vocab=" + configs[i].effectiveBlockVocabSize() + ")");
                } catch (Exception e) {
                    throw new IOException("Failed to load " + STEMS[i]
                            + " from " + modelDir, e);
                }
            }
        } catch (IOException e) {
            manager.close();
            throw e;
        } catch (Exception e) {
            manager.close();
            throw new IOException("Unexpected error loading models from " + modelDir, e);
        }

        // Use the finest model's config/vocab (lod2to1 at index 3)
        BlockVocabulary vocab = BlockVocabulary.fromConfig(configs[3]);
        LOGGER.info("[ProgressiveModelRunner] All 4 models ready.  "
                + "vocab=" + vocab.size() + "  dir=" + modelDir);

        return new ProgressiveModelRunner(manager, models, configs, vocab);
    }

    /**
     * Validate that all files listed in the manifest's {@code required_files}
     * array actually exist in {@code modelDir}.  Fails fast with a clear
     * error listing every missing file so partial deployments are obvious.
     *
     * <p>If the manifest doesn't contain a {@code required_files} key
     * (older exports), falls back to checking the hardcoded STEMS/CONFIG_NAMES.
     */
    private static void validateRequiredFiles(Path modelDir) throws IOException {
        Path manifestPath = modelDir.resolve("pipeline_manifest.json");
        if (!Files.exists(manifestPath)) {
            throw new IOException("pipeline_manifest.json not found in " + modelDir
                    + " — did you forget to deploy it from the export directory?");
        }

        // Parse required_files from the manifest
        String manifestJson = Files.readString(manifestPath);
        List<String> requiredFiles = parseRequiredFiles(manifestJson);

        // Fall back to hardcoded list if manifest predates required_files
        if (requiredFiles.isEmpty()) {
            LOGGER.warning("[ProgressiveModelRunner] Manifest has no 'required_files' — "
                    + "using hardcoded file list (re-export to get automatic validation)");
            requiredFiles = new ArrayList<>();
            for (String stem : STEMS) {
                requiredFiles.add(stem + ".onnx");
            }
            for (String cfg : CONFIG_NAMES) {
                requiredFiles.add(cfg);
            }
        }

        // Check each file
        List<String> missing = new ArrayList<>();
        for (String name : requiredFiles) {
            if (!Files.exists(modelDir.resolve(name))) {
                missing.add(name);
            }
        }

        if (!missing.isEmpty()) {
            StringBuilder sb = new StringBuilder();
            sb.append("Incomplete model deployment in ").append(modelDir).append("!\n");
            sb.append("Missing ").append(missing.size()).append(" required file(s):\n");
            for (String m : missing) {
                sb.append("  - ").append(m).append('\n');
            }
            sb.append("Copy ALL files from the export directory (production/vN/) ");
            sb.append("into ").append(modelDir);
            throw new IOException(sb.toString());
        }

        LOGGER.info("[ProgressiveModelRunner] Deployment validated — "
                + requiredFiles.size() + " required files present in " + modelDir);
    }

    /**
     * Extract the {@code required_files} string array from manifest JSON.
     * Uses simple string parsing to avoid pulling in a JSON library dependency
     * beyond what ConfigLoader already uses.
     */
    private static List<String> parseRequiredFiles(String json) {
        List<String> result = new ArrayList<>();
        int idx = json.indexOf("\"required_files\"");
        if (idx < 0) return result;

        int arrStart = json.indexOf('[', idx);
        if (arrStart < 0) return result;
        int arrEnd = json.indexOf(']', arrStart);
        if (arrEnd < 0) return result;

        String arrContent = json.substring(arrStart + 1, arrEnd);
        // Match quoted strings
        int pos = 0;
        while (pos < arrContent.length()) {
            int q1 = arrContent.indexOf('"', pos);
            if (q1 < 0) break;
            int q2 = arrContent.indexOf('"', q1 + 1);
            if (q2 < 0) break;
            result.add(arrContent.substring(q1 + 1, q2));
            pos = q2 + 1;
        }
        return result;
    }

    // ------------------------------------------------------------------
    // Inference
    // ------------------------------------------------------------------

    /**
     * Run the full 4-stage progressive generation pipeline.
     *
     * <p>Conditioning inputs:
     * <ul>
     *   <li>{@code hp5Row}  — height planes {@code [5][256]} in x-major row-major order
     *       (channel, lx*16+lz), as produced by
     *       {@link com.rhythmatician.lodiffusion.voxy.AnchorSampler#computeHeightPlanes}.</li>
     *   <li>{@code biomeIdx} — canonical biome IDs {@code [16][16]} in [x][z] order
     *       (Minecraft convention); transposed internally to match training.</li>
     *   <li>{@code yIndex}   — raw section Y coordinate (e.g. -4 for y=-64).
     *       The ONNX model clamps to [0, 23] internally, matching training.</li>
     * </ul>
     *
     * @return {@link InferenceResult} with {@code blockLogits}
     *         shaped {@code [1][N][16][16][16]} (2× upsampled from the
     *         pipeline's 8³ native output).  Air is class 0 in the unified
     *         softmax — there is no separate air mask.
     */
    public InferenceResult generate(float[][] hp5Row, int[][] biomeIdx, int yIndex)
            throws TranslateException {

        long t0 = System.currentTimeMillis();
        InferenceBuffers buf = getOrCreateBuffers();

        try (NDManager sub = manager.newSubManager()) {

            // ── Shared conditioning tensors (reuse pooled arrays) ────────
            // hp5: [5][256] x-major  →  [1, 5, 16, 16]  z-major (transpose)
            float[] hpFlat = buf.hpFlat;
            for (int ch = 0; ch < 5; ch++) {
                for (int lx = 0; lx < 16; lx++) {
                    for (int lz = 0; lz < 16; lz++) {
                        hpFlat[ch * 256 + lz * 16 + lx] = hp5Row[ch][lx * 16 + lz];
                    }
                }
            }
            NDArray xHp = sub.create(hpFlat, new Shape(1, 5, 16, 16));

            // biome: [16][16] (x,z)  →  [1, 16, 16]  z-major (transpose)
            long[] bioFlat = buf.bioFlat;
            for (int lx = 0; lx < 16; lx++) {
                for (int lz = 0; lz < 16; lz++) {
                    bioFlat[lz * 16 + lx] = biomeIdx[lx][lz];
                }
            }
            NDArray xBiome = sub.create(bioFlat, new Shape(1, 16, 16));
            NDArray xY     = sub.create(new long[]{yIndex}, new Shape(1));

            // ── Stage 0: init → LOD4  (no parent) ───────────────────────
            NDList s0in   = new NDList(xHp, xBiome, xY);
            NDList s0out  = runStage(0, s0in);
            NDArray logits0 = extractBlockLogits(s0out);  // [1,N,1,1,1]

            // ── Stage 1: LOD4 → LOD3 ────────────────────────────────────
            NDArray p1    = toSolidParentFromLogits(logits0);  // [1,1,1,1,1]
            NDList s1out  = runStage(1, new NDList(xHp, xBiome, xY, p1));
            NDArray logits1 = extractBlockLogits(s1out);       // [1,N,2,2,2]

            // ── Stage 2: LOD3 → LOD2 ────────────────────────────────────
            NDArray p2    = toSolidParentFromLogits(logits1);  // [1,1,2,2,2]
            NDList s2out  = runStage(2, new NDList(xHp, xBiome, xY, p2));
            NDArray logits2 = extractBlockLogits(s2out);       // [1,N,4,4,4]

            // ── Stage 3: LOD2 → LOD1 ────────────────────────────────────
            NDArray p3      = toSolidParentFromLogits(logits2);  // [1,1,4,4,4]
            NDList s3out    = runStage(3, new NDList(xHp, xBiome, xY, p3));
            NDArray logits8 = extractBlockLogits(s3out);         // [1,N,8,8,8]

            // ── Fused extract + 2× upsample into pooled 16³ buffer ──────
            extractAndUpsample5D(logits8, buf.logits16);

            long elapsed = System.currentTimeMillis() - t0;
            return new InferenceResult(buf.logits16, elapsed);
        }
    }

    /**
     * Run a <em>single</em> stage of the progressive pipeline.
     *
     * <p>Designed for the per-stage threading pipeline where each stage runs
     * on its own worker thread.  The conditioning tensors (hp5, biome, y)
     * are reconstructed each call since stages run on different threads with
     * independent {@link NDManager} scopes.
     *
     * <ul>
     *   <li><b>Stages 0-2</b> — returns a deep-copied flat binary parent
     *       array safe for cross-thread handoff.</li>
     *   <li><b>Stage 3</b> — returns the full {@link InferenceResult} with
     *       16³ block logits (referencing thread-local buffers;
     *       must be consumed before the next call on the same thread).</li>
     * </ul>
     *
     * @param stage       which stage to run (0–3)
     * @param hp5Row      height planes {@code [5][256]} in x-major row-major order
     * @param biomeIdx    biome indices {@code [16][16]} in [x][z] order
     * @param yIndex      raw section Y index (e.g. -4 for y=-64)
     * @param parentFlat  binary solid parent from previous stage
     *                    ({@code null} for stage 0)
     * @return stage output with either intermediate parent or final result
     */
    public StageOutput generateStage(int stage, float[][] hp5Row, int[][] biomeIdx,
                                      int yIndex, float[] parentFlat)
            throws TranslateException {

        long t0 = System.currentTimeMillis();
        InferenceBuffers buf = getOrCreateBuffers();

        try (NDManager sub = manager.newSubManager()) {

            // ── Conditioning tensors (same transpose as generate()) ─────
            float[] hpFlat = buf.hpFlat;
            for (int ch = 0; ch < 5; ch++) {
                for (int lx = 0; lx < 16; lx++) {
                    for (int lz = 0; lz < 16; lz++) {
                        hpFlat[ch * 256 + lz * 16 + lx] = hp5Row[ch][lx * 16 + lz];
                    }
                }
            }
            NDArray xHp = sub.create(hpFlat, new Shape(1, 5, 16, 16));

            long[] bioFlat = buf.bioFlat;
            for (int lx = 0; lx < 16; lx++) {
                for (int lz = 0; lz < 16; lz++) {
                    bioFlat[lz * 16 + lx] = biomeIdx[lx][lz];
                }
            }
            NDArray xBiome = sub.create(bioFlat, new Shape(1, 16, 16));
            NDArray xY     = sub.create(new long[]{yIndex}, new Shape(1));

            // ── Build input list ────────────────────────────────────────
            NDList inputs;
            if (stage == 0) {
                inputs = new NDList(xHp, xBiome, xY);
            } else {
                long[] spatialShape = PARENT_INPUT_SHAPES[stage];
                NDArray parent = sub.create(parentFlat,
                        new Shape(1, spatialShape[0], spatialShape[1],
                                  spatialShape[2], spatialShape[3]));
                inputs = new NDList(xHp, xBiome, xY, parent);
            }

            // ── Run the single stage ────────────────────────────────────
            NDList outputs = runStage(stage, inputs);
            long elapsed = System.currentTimeMillis() - t0;

            if (stage < 3) {
                // Intermediate: extract full logits + derive binary solid parent
                NDArray logits = extractBlockLogits(outputs);
                float[] solidFlat = toSolidParentFlatFromLogits(logits);
                float[][][][][] nativeLogits = extract5D(logits);
                return new StageOutput(solidFlat, nativeLogits, null, elapsed);
            } else {
                // Final: extract native 8³ logits + 2× upsample for legacy path
                NDArray logits8 = extractBlockLogits(outputs);
                float[][][][][] nativeLogits = extract5D(logits8);
                extractAndUpsample5D(logits8, buf.logits16);
                return new StageOutput(null, nativeLogits,
                        new InferenceResult(buf.logits16, elapsed),
                        elapsed);
            }
        }
    }

    // ------------------------------------------------------------------
    // Batched inference
    // ------------------------------------------------------------------

    /**
     * Run a <em>single</em> pipeline stage for a <b>batch</b> of sections
     * in one ONNX call.
     *
     * <p>This is the high-throughput counterpart to {@link #generateStage}:
     * instead of processing one section at a time, it stacks N sections'
     * conditioning tensors into batch-N inputs, runs a single ONNX
     * predictor call, then splits the batch output back into per-section
     * {@link StageOutput} objects.
     *
     * <p>Requires ONNX models exported with dynamic batch axes
     * ({@code export_lod.py ≥ v3.1}).  Falls back gracefully to single-sample
     * if called with {@code batchSize == 1}.
     *
     * @param stage        which stage to run (0–3)
     * @param hp5Rows      height planes per section: {@code [batchSize][5][256]}
     * @param biomeIdxs    biome indices per section: {@code [batchSize][16][16]}
     * @param yIndices     raw Y section indices: {@code [batchSize]}
     * @param parentFlats  binary solid parent per section:
     *                     {@code [batchSize][spatialSize]} ({@code null} entries
     *                     allowed for stage 0)
     * @return list of per-section stage outputs, in the same order as the inputs
     */
    public List<StageOutput> generateStageBatch(
            int stage,
            float[][][] hp5Rows,
            int[][][] biomeIdxs,
            int[] yIndices,
            float[][] parentFlats) throws TranslateException {

        int batchSize = hp5Rows.length;
        long t0 = System.currentTimeMillis();

        try (NDManager sub = manager.newSubManager()) {

            // ── Stack conditioning tensors across the batch ─────────────
            // hp: [batchSize, 5, 16, 16]  (transpose x↔z per-sample)
            float[] hpBatch = new float[batchSize * 5 * 256];
            for (int b = 0; b < batchSize; b++) {
                int bOff = b * 5 * 256;
                for (int ch = 0; ch < 5; ch++) {
                    for (int lx = 0; lx < 16; lx++) {
                        for (int lz = 0; lz < 16; lz++) {
                            hpBatch[bOff + ch * 256 + lz * 16 + lx] =
                                    hp5Rows[b][ch][lx * 16 + lz];
                        }
                    }
                }
            }
            NDArray xHp = sub.create(hpBatch, new Shape(batchSize, 5, 16, 16));

            // biome: [batchSize, 16, 16]  (transpose x↔z per-sample)
            long[] bioBatch = new long[batchSize * 256];
            for (int b = 0; b < batchSize; b++) {
                int bOff = b * 256;
                for (int lx = 0; lx < 16; lx++) {
                    for (int lz = 0; lz < 16; lz++) {
                        bioBatch[bOff + lz * 16 + lx] = biomeIdxs[b][lx][lz];
                    }
                }
            }
            NDArray xBiome = sub.create(bioBatch, new Shape(batchSize, 16, 16));

            // y: [batchSize]
            long[] yBatch = new long[batchSize];
            for (int b = 0; b < batchSize; b++) {
                yBatch[b] = yIndices[b];
            }
            NDArray xY = sub.create(yBatch, new Shape(batchSize));

            // ── Build input list ────────────────────────────────────────
            NDList inputs;
            if (stage == 0) {
                inputs = new NDList(xHp, xBiome, xY);
            } else {
                // Stack parents: [batchSize, 1, D, D, D]
                long[] spatialShape = PARENT_INPUT_SHAPES[stage];
                int parentSpatialSize = 1;
                for (long dim : spatialShape) parentSpatialSize *= (int) dim;
                float[] parentBatch = new float[batchSize * parentSpatialSize];
                for (int b = 0; b < batchSize; b++) {
                    System.arraycopy(parentFlats[b], 0,
                            parentBatch, b * parentSpatialSize, parentSpatialSize);
                }
                NDArray parent = sub.create(parentBatch,
                        new Shape(batchSize, spatialShape[0], spatialShape[1],
                                  spatialShape[2], spatialShape[3]));
                inputs = new NDList(xHp, xBiome, xY, parent);
            }

            // ── Single ONNX call for the whole batch ────────────────────
            NDList batchOutputs = runStage(stage, inputs);
            NDArray batchLogits = extractBlockLogits(batchOutputs);
            // batchLogits shape: [batchSize, vocabSize, D, D, D]

            long elapsed = System.currentTimeMillis() - t0;
            long perSample = batchSize > 0 ? elapsed / batchSize : elapsed;

            // ── Convert entire batch to float[] immediately ──────────────
            // OrtNDArray does not support .get(index) or .expandDims();
            // extract the flat array once and slice manually per sample.
            long[] batchShape = batchLogits.getShape().getShape();
            // batchShape: [batchSize, vocabSize, D, D, D]
            int sampleElements = 1;
            for (int d = 1; d < batchShape.length; d++) sampleElements *= (int) batchShape[d];
            float[] allLogitsFlat = batchLogits.toFloatArray();

            // ── Split per-sample results ────────────────────────────────
            List<StageOutput> results = new ArrayList<>(batchSize);
            for (int b = 0; b < batchSize; b++) {
                // Slice this sample's flat data
                float[] sampleFlat = new float[sampleElements];
                System.arraycopy(allLogitsFlat, b * sampleElements, sampleFlat, 0, sampleElements);
                // Wrap as [1, vocabSize, D, D, D] NDArray in the sub-manager
                // (sub is a proper NDManager that supports all operations)
                long[] sampleShape = new long[batchShape.length];
                sampleShape[0] = 1;
                System.arraycopy(batchShape, 1, sampleShape, 1, batchShape.length - 1);
                NDArray singleLogits = sub.create(sampleFlat, new Shape(sampleShape));

                if (stage < 3) {
                    float[] solidFlat = toSolidParentFlatFromLogits(singleLogits);
                    float[][][][][] nativeLogits = extract5D(singleLogits);
                    results.add(new StageOutput(solidFlat, nativeLogits, null, perSample));
                } else {
                    float[][][][][] nativeLogits = extract5D(singleLogits);
                    int vocabSize = (int) batchShape[1];
                    float[][][][][] logits16 = new float[1][vocabSize][16][16][16];
                    extractAndUpsample5D(singleLogits, logits16);
                    results.add(new StageOutput(null, nativeLogits,
                            new InferenceResult(logits16, perSample), perSample));
                }
            }
            return results;
        }
    }

    // ------------------------------------------------------------------
    // Private helpers
    // ------------------------------------------------------------------

    private NDList runStage(int stage, NDList inputs) throws TranslateException {
        try (var predictor = models[stage].newPredictor(new NoopTranslator())) {
            return predictor.predict(inputs);
        }
    }

    /**
     * Derive a binary solid-occupancy parent tensor from block logits.
     *
     * <p>Performs argmax along the channel dimension (dim 1) of the
     * {@code [1, N, D, D, D]} logits tensor.  If the predicted class is 0
     * (air) the parent value is 0; otherwise 1.
     *
     * <p>Returns an NDArray of shape {@code [1, 1, D, D, D]} suitable for
     * feeding as {@code x_parent} to the next pipeline stage.
     */
    private static NDArray toSolidParentFromLogits(NDArray blockLogits) {
        long[] s = blockLogits.getShape().getShape();
        int c = (int) s[1], d = (int) s[2], h = (int) s[3], w = (int) s[4];
        float[] flat = blockLogits.toFloatArray();
        int spatialSize = d * h * w;
        float[] dst = new float[spatialSize];

        // flat layout: [batch=0][channel][d][h][w]  — channel is the fastest-varying outer dim
        for (int i = 0; i < spatialSize; i++) {
            // For each spatial position, find argmax across channels
            int bestCh = 0;
            float bestVal = flat[i];  // channel 0 at spatial position i
            for (int ch = 1; ch < c; ch++) {
                float v = flat[ch * spatialSize + i];
                if (v > bestVal) {
                    bestVal = v;
                    bestCh = ch;
                }
            }
            dst[i] = bestCh == 0 ? 0f : 1f;
        }
        return blockLogits.getManager().create(dst, new Shape(1, 1, d, h, w));
    }

    /**
     * Same as {@link #toSolidParentFromLogits} but returns a fresh
     * {@code float[]} suitable for cross-thread handoff (no NDArray
     * lifecycle dependency).
     */
    private static float[] toSolidParentFlatFromLogits(NDArray blockLogits) {
        long[] s = blockLogits.getShape().getShape();
        int c = (int) s[1], d = (int) s[2], h = (int) s[3], w = (int) s[4];
        float[] flat = blockLogits.toFloatArray();
        int spatialSize = d * h * w;
        float[] dst = new float[spatialSize];

        for (int i = 0; i < spatialSize; i++) {
            int bestCh = 0;
            float bestVal = flat[i];
            for (int ch = 1; ch < c; ch++) {
                float v = flat[ch * spatialSize + i];
                if (v > bestVal) {
                    bestVal = v;
                    bestCh = ch;
                }
            }
            dst[i] = bestCh == 0 ? 0f : 1f;
        }
        return dst;
    }

    /** Extract the block-logits output (the sole output tensor). */
    private static NDArray extractBlockLogits(NDList outputs) {
        if (outputs.size() == 1) return outputs.get(0);
        // Fallback: find by shape (channel dim > 1, rank 5)
        for (NDArray t : outputs) {
            long[] s = t.getShape().getShape();
            if (s.length == 5 && s[1] > 1) return t;
        }
        throw new IllegalStateException(
                "[ProgressiveModelRunner] block_logits not found in outputs");
    }

    /**
     * Extract a 5D NDArray into a Java array at native resolution (no
     * upsampling).  Used to capture intermediate stage block logits for
     * direct-level writes.
     *
     * @param t  NDArray with shape [b, c, d, h, w]
     * @return freshly allocated float[b][c][d][h][w]
     */
    private static float[][][][][] extract5D(NDArray t) {
        long[] s = t.getShape().getShape();
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

    // ------------------------------------------------------------------
    // Thread-local buffer pool (GC pressure reduction)
    // ------------------------------------------------------------------

    /**
     * Pre-allocated arrays for one inference call.  One instance per thread.
     * Eliminates the major GC-pressure sources:
     * <ul>
     *   <li>{@code logits16} — {@code float[1][vocabSize][16][16][16]} (~2.2 MB)</li>
     *   <li>{@code hpFlat} / {@code bioFlat} — conditioning scratch arrays</li>
     * </ul>
     *
     * <p><b>Important:</b> The caller must fully consume the returned
     * {@link InferenceResult} before the next {@code generate()} call on
     * the same thread, because the result references these pooled arrays.
     */
    private static final class InferenceBuffers {
        final int vocabSize;
        final float[] hpFlat  = new float[5 * 16 * 16];
        final long[]  bioFlat = new long[256];
        final float[][][][][] logits16;  // [1][vocabSize][16][16][16]

        InferenceBuffers(int vocabSize) {
            this.vocabSize = vocabSize;
            this.logits16  = new float[1][vocabSize][16][16][16];
        }
    }

    /** Get or create the thread-local inference buffers. */
    private InferenceBuffers getOrCreateBuffers() {
        InferenceBuffers buf = threadBuffers.get();
        int vocabSize = configs[3].effectiveBlockVocabSize();
        if (buf == null || buf.vocabSize != vocabSize) {
            buf = new InferenceBuffers(vocabSize);
            threadBuffers.set(buf);
            LOGGER.info("[ProgressiveModelRunner] Allocated inference buffers "
                    + "for thread " + Thread.currentThread().getName()
                    + " (vocab=" + vocabSize + ", ~"
                    + (vocabSize * 4096 * 4 / 1024 / 1024) + " MB)");
        }
        return buf;
    }

    /**
     * Fused extract + 2× nearest-neighbor upsample.
     *
     * <p>Reads the NDArray's flat data once and writes directly into a
     * pre-allocated {@code [b][c][D*2][D*2][D*2]} output array, performing
     * 2×2×2 replication on the fly.  This eliminates both:
     * <ol>
     *   <li>The intermediate {@code [b][c][D][D][D]} array from extract5D</li>
     *   <li>The second {@code [b][c][D*2][D*2][D*2]} array from upsample2x</li>
     * </ol>
     * saving ~2.5 MB of allocation per call for the logits tensor.
     *
     * @param src  NDArray with shape {@code [b, c, D, D, D]}
     * @param dst  pre-allocated output with shape {@code [b, c, 2D, 2D, 2D]}
     */
    private static void extractAndUpsample5D(NDArray src, float[][][][][] dst) {
        long[] s   = src.getShape().getShape();
        float[] flat = src.toFloatArray();  // only temp allocation (~278 KB for logits)
        int b = (int) s[0], c = (int) s[1];
        int d = (int) s[2], h = (int) s[3], w = (int) s[4];
        int idx = 0;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int di = 0; di < d; di++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++) {
                            float v = flat[idx++];
                            // 2×2×2 nearest-neighbor replication
                            for (int dd = 0; dd < 2; dd++)
                                for (int dh = 0; dh < 2; dh++)
                                    for (int dw = 0; dw < 2; dw++)
                                        dst[bi][ci][di*2+dd][hi*2+dh][wi*2+dw] = v;
                        }
    }

    // ------------------------------------------------------------------
    // Accessors (API compatibility with callers expecting UnifiedModelRunner)
    // ------------------------------------------------------------------

    /** The block vocabulary from the finest model's sidecar config. */
    public BlockVocabulary vocabulary() { return vocabulary; }

    /** ModelConfig from the finest model (lod2to1).  Used for vocab size, etc. */
    public ModelConfig config() { return configs[3]; }

    @Override
    public void close() {
        for (var m : models) {
            if (m != null) try { m.close(); } catch (Exception ignored) {}
        }
        try { manager.close(); } catch (Exception ignored) {}
    }
}
