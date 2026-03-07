package com.rhythmatician.lodiffusion.voxy;

import java.util.logging.Logger;

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

    private static final Logger LOGGER = Logger.getLogger(VoxySectionWriter.class.getName());

    /** Default light value: full sky light, no block light → 0x0F. */
    private static final int DEFAULT_LIGHT = 0x0F;

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

        float[][][][][] logits = result.blockLogits();  // [1][N][16][16][16]
        float[][][][][] mask   = result.airMask();      // [1][1][16][16][16]

        // 1. Create empty VoxelizedSection
        Object section = VoxyCompat.createEmptySection();
        VoxyCompat.setSectionPosition(section, sectionX, sectionY, sectionZ);
        long[] data = VoxyCompat.getSectionData(section);

        int[] nonAirCount = {0};

        // 2. Fill L0 (16³) — data[0..4095]
        for (int y = 0; y < 16; y++) {
            for (int z = 0; z < 16; z++) {
                for (int x = 0; x < 16; x++) {
                    long voxel;

                    if (mask[0][0][x][y][z] <= 0f) {
                        // Air — use air with default light
                        voxel = VoxyCompat.composeVoxel(0, biomeVoxyId, DEFAULT_LIGHT);
                    } else {
                        // Solid — argmax over block logits
                        int bestIdx = 0;
                        float bestVal = logits[0][0][x][y][z];
                        for (int b = 1; b < vocabSize; b++) {
                            float v = logits[0][b][x][y][z];
                            if (v > bestVal) { bestVal = v; bestIdx = b; }
                        }

                        int voxyBlockId = blockMapper.getVoxyBlockId(bestIdx);
                        if (voxyBlockId == 0) {
                            // Mapped to air despite solid mask — keep as air
                            voxel = VoxyCompat.composeVoxel(0, biomeVoxyId, DEFAULT_LIGHT);
                        } else {
                            voxel = VoxyCompat.composeVoxel(voxyBlockId, biomeVoxyId, DEFAULT_LIGHT);
                            nonAirCount[0]++;
                        }
                    }

                    data[VoxyCompat.l0Index(x, y, z)] = voxel;
                }
            }
        }

        VoxyCompat.setNonAirCount(section, nonAirCount[0]);

        // 3. Compute mip pyramid (L1..L4) via Voxy's WorldConversionFactory
        VoxyCompat.mipSection(section, voxyMapper);

        // 4. Push into world
        VoxyCompat.insertUpdate(worldEngine, section);

        LOGGER.fine(() -> "Wrote section (" + sectionX + "," + sectionY + "," + sectionZ
                + ") — " + nonAirCount[0] + " solid voxels");
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
}
