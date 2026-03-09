package com.rhythmatician.lodiffusion.voxy;

import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rhythmatician.lodiffusion.onnx.InferenceResult;

/**
 * Converts model inference output into Voxy {@code VoxelizedSection} objects
 * and pushes them into a Voxy {@code WorldEngine}.
 *
 * <p><b>Pipeline:</b>
 * <ol>
 *   <li>Argmax the block logits → per-voxel model index</li>
 *   <li>Apply air mask (positive logit → solid)</li>
 *   <li>Translate model indices to Voxy block IDs via {@link VoxyBlockMapper}</li>
 *   <li>Pack into 64-bit Voxy voxels and fill a {@code VoxelizedSection}</li>
 *   <li>Compute mip pyramid via {@code WorldConversionFactory.mipSection()}</li>
 *   <li>Inject via {@code WorldUpdater.insertUpdate()}</li>
 * </ol>
 */
public final class VoxySectionWriter {

    private static final Logger LOGGER = LoggerFactory.getLogger(VoxySectionWriter.class);

    /** Default light value: full sky light, no block light → 0x0F. */
    static final int DEFAULT_LIGHT = 0x0F;

    /** Counter for diagnostic logging — log detail for first N sections. Thread-safe. */
    private final AtomicInteger sectionsWritten = new AtomicInteger();

    private final Object worldEngine;
    private final Object voxyMapper;
    private final VoxyBlockMapper blockMapper;

    /**
     * Create a writer for a specific Voxy WorldEngine.
     *
     * @param worldEngine  the Voxy WorldEngine instance (reflected)
     * @param blockMapper  pre-built model→Voxy block ID mapping
     */
    public VoxySectionWriter(Object worldEngine, VoxyBlockMapper blockMapper) {
        this.worldEngine = worldEngine;
        this.voxyMapper  = VoxyCompat.getMapper(worldEngine);
        this.blockMapper = blockMapper;
        LOGGER.info("[VoxySectionWriter] Created — engine={}, mapper={}", 
                worldEngine.getClass().getSimpleName(), voxyMapper.getClass().getSimpleName());
    }

    /**
     * Test constructor for use without a live WorldEngine.
     * Bypasses Voxy runtime requirements.
     *
     * @param blockMapper  model→Voxy block ID mapping (may be a stub)
     */
    public VoxySectionWriter(VoxyBlockMapper blockMapper) {
        this.worldEngine = null;
        this.voxyMapper = null;
        this.blockMapper = blockMapper;
    }

    /**
     * Decode model output and inject it as a Voxy section at the given
     * chunk-section coordinate.
     *
     * @param result        the model's InferenceResult
     * @param vocabSize     number of block types in the model vocabulary
     * @param sectionX      Voxy chunk-section X (block X / 16)
     * @param sectionY      Voxy chunk-section Y (block Y / 16)
     * @param sectionZ      Voxy chunk-section Z (block Z / 16)
     * @param biomeVoxyIds  per-column Voxy biome IDs [16][16], indexed [x][z]
     */
    public void writeSection(InferenceResult result, int vocabSize,
                             int sectionX, int sectionY, int sectionZ,
                             int[][] biomeVoxyIds) {

        // ---- Insert-only guard ----
        // Never overwrite any existing section data.  Each progressive LOD
        // step writes to distinct section coordinates (different resolution
        // grids), so there is no need for self-overwrite tracking.
        if (VoxyCompat.sectionExists(worldEngine, sectionX, sectionY, sectionZ)) {
            if (sectionsWritten.get() < 10) {
                LOGGER.info("[VoxySectionWriter] Skipping ({},{},{}) — section already exists",
                        sectionX, sectionY, sectionZ);
            }
            sectionsWritten.incrementAndGet();
            return;
        }

        boolean detailed = sectionsWritten.get() < 5; // Log detail for first 5 sections

        // Build the filled section
        FilledSectionResult filled = buildFilledSection(result, vocabSize,
                sectionX, sectionY, sectionZ, biomeVoxyIds, detailed);

        Object section = filled.section();
        int nonAirCount = filled.nonAirCount();

        if (detailed) {
            LOGGER.info("[VoxySectionWriter] Section ({},{},{}) — {} non-air voxels out of 4096",
                    sectionX, sectionY, sectionZ, nonAirCount);
        }

        // Short-circuit: if the section is entirely air, skip mip + insert
        if (nonAirCount == 0) {
            if (detailed) {
                LOGGER.warn("[VoxySectionWriter] Skipping all-air section ({},{},{})", sectionX, sectionY, sectionZ);
            }
            sectionsWritten.incrementAndGet();
            return;
        }

        // 3. Compute mip pyramid (L1..L4) via Voxy's WorldConversionFactory
        LOGGER.debug("[VoxySectionWriter] Computing mip pyramid for ({},{},{})", sectionX, sectionY, sectionZ);
        VoxyCompat.mipSection(section, voxyMapper);

        // 4. Push into world
        LOGGER.debug("[VoxySectionWriter] Inserting section ({},{},{}) into Voxy world", sectionX, sectionY, sectionZ);
        VoxyCompat.insertUpdate(worldEngine, section);

        int written = sectionsWritten.incrementAndGet();
        if (detailed || written % 100 == 0) {
            LOGGER.info("[VoxySectionWriter] Wrote section ({},{},{}) — {} solid voxels [total written: {}]",
                    sectionX, sectionY, sectionZ, nonAirCount, written);
        }
    }

    /**
     * Result of building a filled VoxelizedSection (before mip and insert).
     *
     * @param section      the VoxelizedSection object (reflected)
     * @param nonAirCount  count of non-air voxels in L0
     */
    public record FilledSectionResult(Object section, int nonAirCount) {}

    /**
     * Build a filled VoxelizedSection from model output.
     *
     * <p>This method creates the section, sets its position, fills L0 voxels,
     * and returns the result without performing mip computation or insertion.
     *
     * @param result          the model's InferenceResult
     * @param vocabSize       number of block types in the model vocabulary
     * @param sectionX        Voxy chunk-section X (block X / 16)
     * @param sectionY        Voxy chunk-section Y (block Y / 16)
     * @param sectionZ        Voxy chunk-section Z (block Z / 16)
     * @param biomeVoxyIds    per-column Voxy biome IDs [16][16], indexed [x][z]
     * @param logDiagnostics  if true, log air mask statistics
     * @return the filled section and non-air count
     */
    public FilledSectionResult buildFilledSection(InferenceResult result, int vocabSize,
                                            int sectionX, int sectionY, int sectionZ,
                                            int[][] biomeVoxyIds, boolean logDiagnostics) {

        float[][][][][] logits = result.blockLogits();  // [1][N][16][16][16]

        // Diagnostic: check air/solid distribution from argmax
        if (logDiagnostics) {
            int airCount = 0;
            for (int d0 = 0; d0 < 16; d0++)
                for (int d1 = 0; d1 < 16; d1++)
                    for (int d2 = 0; d2 < 16; d2++) {
                        // Quick argmax check: is class 0 (air) the winner?
                        int best = 0;
                        float bestVal = logits[0][0][d0][d1][d2];
                        for (int b = 1; b < vocabSize; b++) {
                            float v = logits[0][b][d0][d1][d2];
                            if (v > bestVal) { bestVal = v; best = b; }
                        }
                        if (best == 0) airCount++;
                    }
            LOGGER.info("[VoxySectionWriter] Section ({},{},{}) argmax stats: air={}/4096, solid={}/4096",
                    sectionX, sectionY, sectionZ, airCount, 4096 - airCount);
        }

        // 1. Create empty VoxelizedSection
        Object section = VoxyCompat.createEmptySection();
        VoxyCompat.setSectionPosition(section, sectionX, sectionY, sectionZ);
        long[] data = VoxyCompat.getSectionData(section);

        int nonAirCount = 0;

        // 2. Fill L0 (16³) — data[0..4095]
        //    Model output dimensions: [batch][channel][d0][d1][d2]
        //    Model axis convention: d0=Y, d1=Z, d2=X (matches Voxy/training)
        //    Voxy l0Index packs as YZX: (y<<8)|(z<<4)|x
        for (int x = 0; x < 16; x++) {
            for (int y = 0; y < 16; y++) {
                for (int z = 0; z < 16; z++) {
                    long voxel;
                    int biome = biomeVoxyIds[x][z]; // per-column biome

                    // Unified argmax over ALL channels (air = class 0)
                    int bestIdx = 0;
                    float bestVal = logits[0][0][y][z][x];
                    for (int b = 1; b < vocabSize; b++) {
                        float v = logits[0][b][y][z][x];
                        if (v > bestVal) {
                            bestVal = v;
                            bestIdx = b;
                        }
                    }

                    if (bestIdx == 0) {
                        // Air — class 0 won the argmax
                        voxel = VoxyCompat.composeVoxel(0, biome, DEFAULT_LIGHT);
                    } else {
                        int voxyBlockId = blockMapper.getVoxyBlockId(bestIdx);
                        if (voxyBlockId == 0) {
                            // Mapped to air despite solid prediction — keep as air
                            voxel = VoxyCompat.composeVoxel(0, biome, DEFAULT_LIGHT);
                        } else {
                            voxel = VoxyCompat.composeVoxel(voxyBlockId, biome, DEFAULT_LIGHT);
                            nonAirCount++;
                        }
                    }

                    data[VoxyCompat.l0Index(x, y, z)] = voxel;
                }
            }
        }

        VoxyCompat.setNonAirCount(section, nonAirCount);

        return new FilledSectionResult(section, nonAirCount);
    }

    /**
     * Write a batch of model results covering a vertical column of sections.
     * Generates 16-block-tall slices from baseY upward.
     *
     * @param result         single 16³ model output
     * @param vocabSize      model vocabulary size
     * @param chunkX         Minecraft chunk X coordinate
     * @param baseY          world Y of the bottom of the 16³ volume
     * @param chunkZ         Minecraft chunk Z coordinate
     * @param biomeVoxyIds   per-column Voxy biome IDs [16][16]
     */
    public void writeChunkSlice(InferenceResult result, int vocabSize,
                                int chunkX, int baseY, int chunkZ,
                                int[][] biomeVoxyIds) {
        // Voxy section coordinates = block / 16 for x,z; block Y / 16 for y
        int sectionX = chunkX;
        int sectionY = baseY / 16;
        int sectionZ = chunkZ;

        writeSection(result, vocabSize, sectionX, sectionY, sectionZ, biomeVoxyIds);
    }

    /**
     * Clear is a no-op with insert-only semantics.
     *
     * <p>Retained for API compatibility.  With the insert-only guard,
     * LODiffusion never overwrites any existing section, so there is
     * nothing to "forget".
     *
     * @param sectionX chunk-section X
     * @param sectionZ chunk-section Z
     * @param baseY    lowest section Y (e.g., -4)
     * @param numY     number of Y sections (e.g., 16)
     */
    public void forgetColumn(int sectionX, int sectionZ, int baseY, int numY) {
        // no-op: insert-only guard handles all protection
    }
}
