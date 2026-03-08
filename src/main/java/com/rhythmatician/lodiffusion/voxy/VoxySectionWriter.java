package com.rhythmatician.lodiffusion.voxy;

import java.util.HashSet;
import java.util.Set;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rhythmatician.lodiffusion.onnx.UnifiedModelRunner.InferenceResult;

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
    private static final int DEFAULT_LIGHT = 0x0F;

    /** Counter for diagnostic logging — log detail for first N sections. */
    private int sectionsWritten = 0;

    /**
     * Tracks section positions we have written this session.
     * Used to distinguish our LODiffusion data (safe to overwrite during
     * progressive refinement) from Voxy-native data (must not overwrite).
     */
    private final Set<Long> writtenSections = new HashSet<>();

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
     * Decode model output and inject it as a Voxy section at the given
     * chunk-section coordinate.
     *
     * @param result      the model's InferenceResult
     * @param vocabSize   number of block types in the model vocabulary
     * @param sectionX    Voxy chunk-section X (block X / 16)
     * @param sectionY    Voxy chunk-section Y (block Y / 16)
     * @param sectionZ    Voxy chunk-section Z (block Z / 16)
     * @param biomeVoxyId Voxy biome ID to use for all voxels in this section
     */
    public void writeSection(InferenceResult result, int vocabSize,
                             int sectionX, int sectionY, int sectionZ,
                             int biomeVoxyId) {

        // ---- Overwrite protection ----
        // Only protect Voxy-native data (from real chunk loading).
        // Our own LODiffusion writes are safe to overwrite — that's how
        // progressive refinement works (LOD 4 → 3 → 2 → 1).
        long posKey = sectionPosKey(sectionX, sectionY, sectionZ);
        if (VoxyCompat.sectionExists(worldEngine, sectionX, sectionY, sectionZ)
                && !writtenSections.contains(posKey)) {
            if (sectionsWritten < 10) {
                LOGGER.info("[VoxySectionWriter] Skipping ({},{},{}) — Voxy has native data",
                        sectionX, sectionY, sectionZ);
            }
            sectionsWritten++;
            return;
        }

        float[][][][][] logits = result.blockLogits();  // [1][N][16][16][16]
        float[][][][][] mask   = result.airMask();      // [1][1][16][16][16]

        boolean detailed = sectionsWritten < 5; // Log detail for first 5 sections

        // Diagnostic: check air mask stats
        if (detailed) {
            int positiveCount = 0;
            float minMask = Float.MAX_VALUE, maxMask = -Float.MAX_VALUE;
            for (int d0 = 0; d0 < 16; d0++)
                for (int d1 = 0; d1 < 16; d1++)
                    for (int d2 = 0; d2 < 16; d2++) {
                        float v = mask[0][0][d0][d1][d2];
                        if (v > 0) positiveCount++;
                        if (v < minMask) minMask = v;
                        if (v > maxMask) maxMask = v;
                    }
            LOGGER.info("[VoxySectionWriter] Section ({},{},{}) air_mask stats: positive={}/4096, min={}, max={}",
                    sectionX, sectionY, sectionZ, positiveCount, minMask, maxMask);
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

                    // Access model output at [batch=0][chan][d0=Y][d1=Z][d2=X]
                    if (mask[0][0][y][z][x] <= 0f) {
                        // Air — use air with default light
                        voxel = VoxyCompat.composeVoxel(0, biomeVoxyId, DEFAULT_LIGHT);
                    } else {
                        // Solid — argmax over block logits
                        int bestIdx = 0;
                        float bestVal = logits[0][0][y][z][x];
                        for (int b = 1; b < vocabSize; b++) {
                            float v = logits[0][b][y][z][x];
                            if (v > bestVal) { bestVal = v; bestIdx = b; }
                        }

                        int voxyBlockId = blockMapper.getVoxyBlockId(bestIdx);
                        if (voxyBlockId == 0) {
                            // Mapped to air despite solid mask — keep as air
                            voxel = VoxyCompat.composeVoxel(0, biomeVoxyId, DEFAULT_LIGHT);
                        } else {
                            voxel = VoxyCompat.composeVoxel(voxyBlockId, biomeVoxyId, DEFAULT_LIGHT);
                            nonAirCount++;
                        }
                    }

                    data[VoxyCompat.l0Index(x, y, z)] = voxel;
                }
            }
        }

        VoxyCompat.setNonAirCount(section, nonAirCount);

        if (detailed) {
            LOGGER.info("[VoxySectionWriter] Section ({},{},{}) — {} non-air voxels out of 4096",
                    sectionX, sectionY, sectionZ, nonAirCount);
        }

        // Short-circuit: if the section is entirely air, skip mip + insert
        if (nonAirCount == 0) {
            if (detailed) {
                LOGGER.warn("[VoxySectionWriter] Skipping all-air section ({},{},{})", sectionX, sectionY, sectionZ);
            }
            sectionsWritten++;
            return;
        }

        // 3. Compute mip pyramid (L1..L4) via Voxy's WorldConversionFactory
        LOGGER.debug("[VoxySectionWriter] Computing mip pyramid for ({},{},{})", sectionX, sectionY, sectionZ);
        VoxyCompat.mipSection(section, voxyMapper);

        // 4. Push into world
        LOGGER.debug("[VoxySectionWriter] Inserting section ({},{},{}) into Voxy world", sectionX, sectionY, sectionZ);
        VoxyCompat.insertUpdate(worldEngine, section);

        // Track that we wrote this section (for progressive overwrite)
        writtenSections.add(posKey);

        if (detailed || sectionsWritten % 100 == 0) {
            LOGGER.info("[VoxySectionWriter] Wrote section ({},{},{}) — {} solid voxels [total written: {}]",
                    sectionX, sectionY, sectionZ, nonAirCount, sectionsWritten + 1);
        }
        sectionsWritten++;
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
     * @param biomeVoxyId    Voxy biome ID
     */
    public void writeChunkSlice(InferenceResult result, int vocabSize,
                                int chunkX, int baseY, int chunkZ,
                                int biomeVoxyId) {
        // Voxy section coordinates = block / 16 for x,z; block Y / 16 for y
        int sectionX = chunkX;
        int sectionY = baseY / 16;
        int sectionZ = chunkZ;

        writeSection(result, vocabSize, sectionX, sectionY, sectionZ, biomeVoxyId);
    }

    /**
     * Clear ownership claims for an entire column of sections.
     *
     * <p>Called when real vanilla chunks have been loaded at this position.
     * This ensures that even if the chunk later unloads, we won't overwrite
     * Voxy's native data on a subsequent LOD pass.
     *
     * @param sectionX chunk-section X
     * @param sectionZ chunk-section Z
     * @param baseY    lowest section Y (e.g., -4)
     * @param numY     number of Y sections (e.g., 16)
     */
    public void forgetColumn(int sectionX, int sectionZ, int baseY, int numY) {
        for (int sy = baseY; sy < baseY + numY; sy++) {
            writtenSections.remove(sectionPosKey(sectionX, sy, sectionZ));
        }
    }

    /** Compact key for a section position (no LOD component). */
    private static long sectionPosKey(int x, int y, int z) {
        return ((long) (x & 0xFFFF) << 32)
             | ((long) (y & 0xFFFF) << 16)
             | (z & 0xFFFFL);
    }
}
