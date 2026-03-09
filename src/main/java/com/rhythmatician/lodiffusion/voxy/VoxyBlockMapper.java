package com.rhythmatician.lodiffusion.voxy;

import java.lang.reflect.Method;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rhythmatician.lodiffusion.onnx.BlockVocabulary;

import net.minecraft.block.BlockState;

/**
 * Translates model vocabulary indices → Voxy internal block IDs,
 * and canonical biome IDs → Voxy internal biome IDs.
 *
 * <p>Voxy assigns its own numeric IDs to {@link BlockState}s and biomes via
 * its {@code Mapper} class.  This bridge pre-resolves every entry so that
 * output decoding is a simple array lookup.
 */
public final class VoxyBlockMapper {

    private static final Logger LOGGER = LoggerFactory.getLogger(VoxyBlockMapper.class);

    private final int[] modelIndexToVoxyId;       // [vocabSize]  model output idx → Voxy blockId
    private final int[] canonicalBiomeToVoxyId;   // [BiomeMapping.size()]  canonical biome → Voxy biomeId
    private final int defaultBiomeVoxyId;         // Voxy biome ID for "minecraft:plains"

    private VoxyBlockMapper(int[] modelIndexToVoxyId, int defaultBiomeVoxyId,
                            int[] canonicalBiomeToVoxyId) {
        this.modelIndexToVoxyId = modelIndexToVoxyId;
        this.defaultBiomeVoxyId = defaultBiomeVoxyId;
        this.canonicalBiomeToVoxyId = canonicalBiomeToVoxyId;
    }

    /**
     * Build the mapping by registering every BlockState with Voxy's Mapper,
     * and resolving canonical biome IDs to Voxy biome IDs.
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
                if (i < 20 || voxyId == 0 && i < 100) {
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

            // Resolve canonical biome IDs → Voxy biome IDs via getBiomeEntries()
            int[] biomeMappings = resolveBiomeMappings(voxyMapper);
            int plainsCanonical = BiomeMapping.toCanonicalId("minecraft:plains");
            int defaultBiome = biomeMappings[plainsCanonical];

            return new VoxyBlockMapper(mapping, defaultBiome, biomeMappings);

        } catch (Exception e) {
            throw new RuntimeException("Failed to build VoxyBlockMapper", e);
        }
    }

    /**
     * Resolve canonical biome IDs to Voxy internal biome IDs.
     *
     * <p>Queries Voxy's {@code Mapper.getBiomeEntries()} which returns all
     * biomes that Voxy has already registered.  Each entry has a string name
     * (e.g. {@code "minecraft:plains"}) and an integer ID.  We match these
     * against our canonical biome names to build the translation table.
     *
     * <p>Biomes not yet registered in Voxy will map to 0 (the first biome
     * Voxy registered, usually the biome at spawn).
     */
    private static int[] resolveBiomeMappings(Object voxyMapper) {
        int[] map = new int[BiomeMapping.size()];
        try {
            // Reflect Mapper.getBiomeEntries() → BiomeEntry[]
            Method getBiomeEntries = voxyMapper.getClass().getMethod("getBiomeEntries");
            Object[] entries = (Object[]) getBiomeEntries.invoke(voxyMapper);

            if (entries == null || entries.length == 0) {
                LOGGER.warn("[VoxyBlockMapper] No biome entries from Voxy — all biomes map to 0");
                return map;
            }

            // For each Voxy BiomeEntry, match to our canonical ID
            int resolved = 0;
            for (Object entry : entries) {
                String biomeName = (String) entry.getClass().getField("biome").get(entry);
                int voxyId = entry.getClass().getField("id").getInt(entry);

                int canonicalId = BiomeMapping.toCanonicalId(biomeName);
                if (canonicalId != BiomeMapping.UNKNOWN_BIOME_ID) {
                    map[canonicalId] = voxyId;
                    resolved++;
                }
            }

            LOGGER.info("[VoxyBlockMapper] Resolved {}/{} canonical biomes from {} Voxy entries",
                    resolved, BiomeMapping.size(), entries.length);

        } catch (Exception e) {
            LOGGER.warn("[VoxyBlockMapper] Failed to resolve biome mappings: {} — " +
                    "all biomes will use default", e.getMessage());
        }
        return map;
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

    /**
     * Get the Voxy biome ID for a canonical biome index.
     *
     * @param canonicalBiomeIdx canonical biome ID (0–53 from {@link BiomeMapping})
     * @return Voxy internal biome ID, or {@link #defaultBiomeVoxyId()} for
     *         out-of-range or unknown biomes
     */
    public int getVoxyBiomeId(int canonicalBiomeIdx) {
        if (canonicalBiomeIdx < 0 || canonicalBiomeIdx >= canonicalBiomeToVoxyId.length) {
            return defaultBiomeVoxyId;
        }
        return canonicalBiomeToVoxyId[canonicalBiomeIdx];
    }
}
