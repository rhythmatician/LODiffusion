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
 * <pre>
 * Pipeline stages:
 *   init_to_lod4        inputs: (hp[1,5,16,16], biome[1,16,16], y[1])
 *                       output: block_logits[1,N,1,1,1], air_mask[1,1,1,1,1]
 *
 *   refine_lod4_to_lod3 inputs: (hp, biome, y, parent[1,1,1,1,1])
 *                       output: block_logits[1,N,2,2,2], air_mask[1,1,2,2,2]
 *
 *   refine_lod3_to_lod2 inputs: (hp, biome, y, parent[1,1,2,2,2])
 *                       output: block_logits[1,N,4,4,4], air_mask[1,1,4,4,4]
 *
 *   refine_lod2_to_lod1 inputs: (hp, biome, y, parent[1,1,4,4,4])
 *                       output: block_logits[1,N,8,8,8], air_mask[1,1,8,8,8]
 * </pre>
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

    private final NDManager manager;
    @SuppressWarnings("unchecked")
    private final ZooModel<NDList, NDList>[] models;
    private final ModelConfig[]              configs;
    private final BlockVocabulary            vocabulary;

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
     * @return {@link InferenceResult} with {@code blockLogits} and {@code airMask}
     *         both shaped {@code [1][N/1][16][16][16]} (2× upsampled from the
     *         pipeline's 8³ native output)
     */
    public InferenceResult generate(float[][] hp5Row, int[][] biomeIdx, int yIndex)
            throws TranslateException {

        long t0 = System.currentTimeMillis();

        try (NDManager sub = manager.newSubManager()) {

            // ── Shared conditioning tensors ──────────────────────────────
            // hp5: [5][256] x-major  →  [1, 5, 16, 16]  z-major (transpose)
            // Training used (z, x) layout (produced by parse_heightmap().T in Python)
            float[] hpFlat = new float[5 * 16 * 16];
            for (int ch = 0; ch < 5; ch++) {
                for (int lx = 0; lx < 16; lx++) {
                    for (int lz = 0; lz < 16; lz++) {
                        // src: x-major ch*256 + lx*16+lz
                        // dst: z-major ch*256 + lz*16+lx
                        hpFlat[ch * 256 + lz * 16 + lx] = hp5Row[ch][lx * 16 + lz];
                    }
                }
            }
            NDArray xHp = sub.create(hpFlat, new Shape(1, 5, 16, 16));

            // biome: [16][16] (x,z)  →  [1, 16, 16]  z-major (transpose)
            long[] bioFlat = new long[256];
            for (int lx = 0; lx < 16; lx++) {
                for (int lz = 0; lz < 16; lz++) {
                    bioFlat[lz * 16 + lx] = biomeIdx[lx][lz];
                }
            }
            NDArray xBiome = sub.create(bioFlat, new Shape(1, 16, 16));
            NDArray xY     = sub.create(new long[]{yIndex}, new Shape(1));

            // ── Stage 0: init → LOD4  (no parent) ───────────────────────
            NDList s0in  = new NDList(xHp, xBiome, xY);
            NDList s0out = runStage(0, s0in);
            NDArray air0 = extractAirMask(s0out);   // [1,1,1,1,1]

            // ── Stage 1: LOD4 → LOD3 ────────────────────────────────────
            NDArray p1   = toSolidParent(air0);     // [1,1,1,1,1] binary
            NDList s1out = runStage(1, new NDList(xHp, xBiome, xY, p1));
            NDArray air1 = extractAirMask(s1out);   // [1,1,2,2,2]

            // ── Stage 2: LOD3 → LOD2 ────────────────────────────────────
            NDArray p2   = toSolidParent(air1);     // [1,1,2,2,2] binary
            NDList s2out = runStage(2, new NDList(xHp, xBiome, xY, p2));
            NDArray air2 = extractAirMask(s2out);   // [1,1,4,4,4]

            // ── Stage 3: LOD2 → LOD1 ────────────────────────────────────
            NDArray p3      = toSolidParent(air2);  // [1,1,4,4,4] binary
            NDList s3out    = runStage(3, new NDList(xHp, xBiome, xY, p3));
            NDArray logits8 = extractBlockLogits(s3out);  // [1,N,8,8,8]
            NDArray air8    = extractAirMask(s3out);      // [1,1,8,8,8]

            // ── 2× nearest-neighbor upsample: 8³ → 16³ ──────────────────
            float[][][][][] bl8  = extract5D(logits8);
            float[][][][][] am8  = extract5D(air8);
            float[][][][][] bl16 = upsample2x(bl8);
            float[][][][][] am16 = upsample2x(am8);

            long elapsed = System.currentTimeMillis() - t0;
            return new InferenceResult(bl16, am16, elapsed);
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
     * Convert raw air-mask logits to a binary solid-occupancy parent tensor.
     * Model convention: positive logit = solid; negative = air.
     *
     * <p>Note: DJL's {@code OrtNDArray.gt()} triggers infinite recursion in
     * {@code NDArrayAdapter.gt()} (StackOverflowError), so we threshold
     * manually via the raw float array.
     */
    private static NDArray toSolidParent(NDArray airLogits) {
        float[] src = airLogits.toFloatArray();
        float[] dst = new float[src.length];
        for (int i = 0; i < src.length; i++) {
            dst[i] = src[i] > 0f ? 1f : 0f;
        }
        return airLogits.getManager().create(dst, airLogits.getShape());
    }

    /** Extract the air-mask output (channel dim == 1). */
    private static NDArray extractAirMask(NDList outputs) {
        for (NDArray t : outputs) {
            long[] s = t.getShape().getShape();
            if (s.length == 5 && s[1] == 1) return t;
        }
        throw new IllegalStateException(
                "[ProgressiveModelRunner] air_mask (ch=1, rank=5) not found in outputs");
    }

    /** Extract the block-logits output (channel dim > 1). */
    private static NDArray extractBlockLogits(NDList outputs) {
        for (NDArray t : outputs) {
            long[] s = t.getShape().getShape();
            if (s.length == 5 && s[1] > 1) return t;
        }
        throw new IllegalStateException(
                "[ProgressiveModelRunner] block_logits (ch>1, rank=5) not found in outputs");
    }

    /** Copy a rank-5 NDArray into a Java {@code float[][][][][]}. */
    private static float[][][][][] extract5D(NDArray t) {
        long[] s    = t.getShape().getShape();
        float[] src = t.toFloatArray();
        int b = (int) s[0], c = (int) s[1], d = (int) s[2],
            h = (int) s[3], w = (int) s[4];
        float[][][][][] out = new float[b][c][d][h][w];
        int idx = 0;
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int di = 0; di < d; di++)
                    for (int hi = 0; hi < h; hi++)
                        for (int wi = 0; wi < w; wi++)
                            out[bi][ci][di][hi][wi] = src[idx++];
        return out;
    }

    /**
     * 2× nearest-neighbor upsample along the spatial dimensions [d, h, w].
     * Each voxel is replicated into a 2×2×2 block.
     */
    private static float[][][][][] upsample2x(float[][][][][] t) {
        int b = t.length, c = t[0].length, D = t[0][0].length;
        float[][][][][] out = new float[b][c][D * 2][D * 2][D * 2];
        for (int bi = 0; bi < b; bi++)
            for (int ci = 0; ci < c; ci++)
                for (int di = 0; di < D; di++)
                    for (int hi = 0; hi < D; hi++)
                        for (int wi = 0; wi < D; wi++) {
                            float v = t[bi][ci][di][hi][wi];
                            for (int dd = 0; dd < 2; dd++)
                                for (int dh = 0; dh < 2; dh++)
                                    for (int dw = 0; dw < 2; dw++)
                                        out[bi][ci][di*2+dd][hi*2+dh][wi*2+dw] = v;
                        }
        return out;
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
