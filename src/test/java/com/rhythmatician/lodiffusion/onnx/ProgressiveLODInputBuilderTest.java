package com.rhythmatician.lodiffusion.onnx;

import static org.junit.jupiter.api.Assertions.assertDoesNotThrow;
import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotNull;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumMap;
import java.util.EnumSet;
import java.util.HashMap;
import java.util.Map;

import org.junit.jupiter.api.Test;

import com.rhythmatician.lodiffusion.world.noise.NoiseTap;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import net.minecraft.world.Heightmap;

public class ProgressiveLODInputBuilderTest {

    @Test
    void buildInputs_forInitStage_hasExpectedShapes() throws Exception {
        Path cfgPath = Path.of("onnx_export", "model_config_init.json");
        assertTrue(Files.exists(cfgPath), "Missing model_config_init.json");

        ModelConfig cfg = ConfigLoader.load(cfgPath);
        try (NDManager manager = NDManager.newBaseManager()) {
            ProgressiveLODInputBuilder ib = new ProgressiveLODInputBuilder(manager, cfg);
            NoiseTap.Cache cache = syntheticCache();

            Map<String, NDArray> inputs = ib.buildInputs(cache, null, /*lodLevel*/0);

            // Required inputs
            assertShape(inputs.get("x_parent_prev"), new Shape(1, 1, 1, 1, 1));
            assertShape(inputs.get("x_height_planes"), new Shape(1, 5, 1, 16, 16));
            assertShape(inputs.get("x_biome_quart"), new Shape(1, 6, 4, 4, 4));
            assertShape(inputs.get("x_router6"), new Shape(1, 6, 1, 16, 16));
            assertShape(inputs.get("x_chunk_pos"), new Shape(1, 2));
            assertShape(inputs.get("x_lod"), new Shape(1, 1));

            // Optional inputs if declared by config
            if (cfg.isOptionalInput("x_barrier")) {
                assertShape(inputs.get("x_barrier"), new Shape(1, 1, 1, 16, 16));
            }
            if (cfg.isOptionalInput("x_aquifer3")) {
                assertShape(inputs.get("x_aquifer3"), new Shape(1, 3, 1, 16, 16));
            }
            if (cfg.isOptionalInput("x_cave_prior4")) {
                assertShape(inputs.get("x_cave_prior4"), new Shape(1, 1, 4, 4, 4));
            }
        }
    }

    @Test
    void buildInputs_forLod1to0_hasExpectedParentShape() throws Exception {
        Path cfgPath = Path.of("onnx_export", "model_config_lod1_to_lod0.json");
        assertTrue(Files.exists(cfgPath), "Missing model_config_lod1_to_lod0.json");

        ModelConfig cfg = ConfigLoader.load(cfgPath);
        try (NDManager manager = NDManager.newBaseManager()) {
            ProgressiveLODInputBuilder ib = new ProgressiveLODInputBuilder(manager, cfg);
            NoiseTap.Cache cache = syntheticCache();

            // Provide a non-null parent from previous stage (8^3)
            NDArray parent = manager.zeros(new Shape(1, 1, 8, 8, 8));
            Map<String, NDArray> inputs = ib.buildInputs(cache, parent, /*lodLevel*/4);

            assertShape(inputs.get("x_parent_prev"), new Shape(1, 1, 8, 8, 8));
            assertShape(inputs.get("x_height_planes"), new Shape(1, 5, 1, 16, 16));
            assertShape(inputs.get("x_biome_quart"), new Shape(1, 6, 4, 4, 4));
            assertShape(inputs.get("x_router6"), new Shape(1, 6, 1, 16, 16));
            assertShape(inputs.get("x_chunk_pos"), new Shape(1, 2));
            assertShape(inputs.get("x_lod"), new Shape(1, 1));
        }
    }

    @Test
    void config_files_validate() throws Exception {
        // Load all known config files and call validate()
        String[] files = new String[] {
            "model_config_init.json",
            "model_config_lod4_to_lod3.json",
            "model_config_lod3_to_lod2.json",
            "model_config_lod2_to_lod1.json",
            "model_config_lod1_to_lod0.json"
        };
        for (String f : files) {
            Path cfgPath = Path.of("onnx_export", f);
            assertTrue(Files.exists(cfgPath), "Missing config: " + f);
            ModelConfig cfg = ConfigLoader.load(cfgPath);
            assertNotNull(cfg);
            assertDoesNotThrow(cfg::validate, "Validation failed for " + f);
            assertTrue(cfg.getOutputResolution() > 0);
        }
    }

    private static void assertShape(NDArray a, Shape expected) {
        assertNotNull(a, "NDArray was null");
        assertEquals(expected, a.getShape(), "Shape mismatch");
    }

    private static NoiseTap.Cache syntheticCache() {
        // Router maps with required fields at 16x16x16
        Map<NoiseTap.RouterField, float[][][]> router = new EnumMap<>(NoiseTap.RouterField.class);

        // Helper to create a field filled with simple ramp for determinism
        for (NoiseTap.RouterField f : EnumSet.of(
                NoiseTap.RouterField.TEMPERATURE,
                NoiseTap.RouterField.VEGETATION,
                NoiseTap.RouterField.CONTINENTS,
                NoiseTap.RouterField.EROSION,
                NoiseTap.RouterField.DEPTH,
                NoiseTap.RouterField.RIDGES,
                NoiseTap.RouterField.BARRIER,
                NoiseTap.RouterField.FLUID_FLOODEDNESS,
                NoiseTap.RouterField.FLUID_SPREAD,
                NoiseTap.RouterField.LAVA,
                NoiseTap.RouterField.INITIAL_DENSITY_NO_JAG)) {
            router.put(f, field3D());
        }

        // Biomes 4x4x4
        int[][][] biomes4 = new int[4][4][4];
        for (int x = 0; x < 4; x++)
            for (int z = 0; z < 4; z++)
                for (int y = 0; y < 4; y++)
                    biomes4[x][z][y] = (x + z + y) % 20;

        // Heightmaps 16x16
        Map<Heightmap.Type, short[][]> heightmaps = new HashMap<>();
        heightmaps.put(Heightmap.Type.WORLD_SURFACE_WG, heightmap2D((short)64));
        heightmaps.put(Heightmap.Type.OCEAN_FLOOR_WG, heightmap2D((short)50));

        return new NoiseTap.Cache(router, biomes4, heightmaps, 0, 384, 0, 0, 1234L);
    }

    private static float[][][] field3D() {
        float[][][] out = new float[16][16][16];
        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++)
                for (int y = 0; y < 16; y++)
                    out[x][z][y] = x + z + y;
        return out;
    }

    private static short[][] heightmap2D(short base) {
        short[][] hm = new short[16][16];
        for (int x = 0; x < 16; x++)
            for (int z = 0; z < 16; z++)
                hm[x][z] = (short)(base + (x - z));
        return hm;
    }
}
