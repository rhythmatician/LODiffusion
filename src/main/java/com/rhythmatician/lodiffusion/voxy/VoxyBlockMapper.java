package com.rhythmatician.lodiffusion.voxy;

import java.lang.reflect.Method;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rhythmatician.lodiffusion.onnx.BlockVocabulary;

import net.minecraft.block.BlockState;

/**
 * Translates model vocabulary indices → Voxy internal block IDs.
 *
 * <p>Voxy assigns its own numeric IDs to {@link BlockState}s via its
 * {@code Mapper} class.  This bridge pre-resolves every entry in our
 * {@link BlockVocabulary} to the corresponding Voxy ID so that output
 * decoding is a simple array lookup.
 */
public final class VoxyBlockMapper {

    private static final Logger LOGGER = LoggerFactory.getLogger(VoxyBlockMapper.class);

    private final int[] modelIndexToVoxyId;   // [vocabSize]  model output idx → Voxy blockId
    private final int defaultBiomeVoxyId;     // Voxy biome ID for "minecraft:plains"

    private VoxyBlockMapper(int[] modelIndexToVoxyId, int defaultBiomeVoxyId) {
        this.modelIndexToVoxyId = modelIndexToVoxyId;
        this.defaultBiomeVoxyId = defaultBiomeVoxyId;
    }

    /**
     * Build the mapping by registering every BlockState with Voxy's Mapper.
     *
     * @param vocab       our model's BlockVocabulary
     * @param voxyMapper  the Voxy Mapper object (obtained via {@link VoxyCompat#getMapper})
     */
    public static VoxyBlockMapper build(BlockVocabulary vocab, Object voxyMapper) {
        try {
            Method getIdMethod = voxyMapper.getClass().getMethod("getIdForBlockState",
                    BlockState.class);

            int[] mapping = new int[vocab.size()];
            int resolved = 0;
            int zeroMapped = 0; // Track how many resolve to 0 (air)

            for (int i = 0; i < vocab.size(); i++) {
                BlockState state = vocab.getState(i);
                int voxyId = (int) getIdMethod.invoke(voxyMapper, state);
                mapping[i] = voxyId;
                if (voxyId >= 0) resolved++;
                if (voxyId == 0) zeroMapped++;

                // Log first 20 entries and any that map to 0 (air)
                if (i < 20 || (voxyId == 0 && i < 100)) {
                    LOGGER.info("[VoxyBlockMapper] idx={} name='{}' → voxyId={}",
                            i, vocab.getName(i), voxyId);
                }
            }

            LOGGER.info("[VoxyBlockMapper] {} / {} model indices mapped to Voxy IDs " +
                    "({} mapped to 0=air)", resolved, vocab.size(), zeroMapped);

            // Check a few key blocks
            for (int i = 0; i < Math.min(vocab.size(), 10); i++) {
                LOGGER.info("[VoxyBlockMapper] Summary idx={}: '{}' → voxyId={}",
                        i, vocab.getName(i), mapping[i]);
            }

            // Default biome — we'll register plains for now
            int defaultBiome = 0; // Will be set per-column at injection time

            return new VoxyBlockMapper(mapping, defaultBiome);

        } catch (Exception e) {
            throw new RuntimeException("Failed to build VoxyBlockMapper", e);
        }
    }

    /**
     * Get the Voxy block ID for a model output index.
     * Returns 0 (air) for out-of-range indices.
     */
    public int getVoxyBlockId(int modelIndex) {
        if (modelIndex < 0 || modelIndex >= modelIndexToVoxyId.length) return 0;
        return modelIndexToVoxyId[modelIndex];
    }

    /**
     * Compose a full 64-bit Voxy voxel from a model block index and biome.
     *
     * @param modelBlockIndex  index from the model's argmax output
     * @param voxyBiomeId      Voxy biome ID (from Mapper.getIdForBiome)
     * @param light            packed light value (blockLight << 4 | skyLight)
     */
    public long composeVoxel(int modelBlockIndex, int voxyBiomeId, int light) {
        int blockId = getVoxyBlockId(modelBlockIndex);
        return VoxyCompat.composeVoxel(blockId, voxyBiomeId, light);
    }

    /** Number of entries in the mapping table. */
    public int size() {
        return modelIndexToVoxyId.length;
    }

    /** The default biome Voxy ID (currently plains). */
    public int defaultBiomeVoxyId() {
        return defaultBiomeVoxyId;
    }
}
