package com.rhythmatician.lodiffusion.command;

import java.io.IOException;
import java.lang.reflect.Field;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.EnumSet;
import java.util.logging.Logger;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.context.CommandContext;
import com.rhythmatician.lodiffusion.HelloTerrainMod;
import com.rhythmatician.lodiffusion.world.noise.NoiseTap;
import com.rhythmatician.lodiffusion.world.noise.NoiseTap.RouterField;

import net.minecraft.command.permission.Permission;
import net.minecraft.command.permission.PermissionLevel;
import net.minecraft.registry.entry.RegistryEntry;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;
import net.minecraft.util.math.ChunkPos;
import net.minecraft.world.Heightmap;
import net.minecraft.world.biome.Biome;
import net.minecraft.world.biome.source.BiomeAccess;
import net.minecraft.world.chunk.Chunk;
import net.minecraft.world.chunk.ChunkStatus;
import net.minecraft.world.gen.chunk.ChunkGenerator;
import net.minecraft.world.gen.chunk.NoiseChunkGenerator;
import net.minecraft.world.gen.noise.NoiseConfig;

/**
 * Server command {@code /dumpnoise <radius>} that extracts real vanilla noise
 * signals from loaded chunks using {@link NoiseTap} and serialises them to
 * JSON files under {@code run/noise_dumps/}.
 *
 * <p>Each dump file contains:
 * <ul>
 *   <li>{@code router6} — 6×16×16×16 float CORE router (temperature, vegetation,
 *       continents, erosion, depth, ridges) flattened to 6 channels × 256 per Y-slice</li>
 *   <li>{@code heightmap_surface} — 16×16 WORLD_SURFACE_WG heights</li>
 *   <li>{@code heightmap_ocean_floor} — 16×16 OCEAN_FLOOR_WG heights</li>
 *   <li>{@code biomes} — 4×4×4 biome lattice IDs</li>
 *   <li>{@code seed}, {@code chunk_x}, {@code chunk_z}</li>
 * </ul>
 *
 * <p>Usage: {@code /dumpnoise [radius]}  (default radius = 8 chunks)
 *
 * <p>The output JSON can be consumed by the Python training pipeline through
 * {@code scripts/extraction/chunk_extractor.py} when the
 * {@code --noise-dump-dir} option is supplied.
 */
public final class NoiseDumperCommand {

    private static final Logger LOG = Logger.getLogger(NoiseDumperCommand.class.getName());

    /** CORE router fields — matches Python training order. */
    private static final EnumSet<RouterField> CORE_FIELDS = NoiseTap.getTierFields(
            NoiseTap.PerformanceTier.CORE);

    private static final EnumSet<Heightmap.Type> HEIGHTMAPS = EnumSet.of(
            Heightmap.Type.WORLD_SURFACE_WG,
            Heightmap.Type.OCEAN_FLOOR_WG);

    private NoiseDumperCommand() {}

    /**
     * Register {@code /dumpnoise [radius]} with the Brigadier dispatcher.
     * Should be called from {@link com.rhythmatician.lodiffusion.HelloTerrainMod}.
     */
    public static void register(CommandDispatcher<ServerCommandSource> dispatcher) {
        dispatcher.register(CommandManager.literal("dumpnoise")
            .requires(src -> src.getPermissions().hasPermission(
                    new Permission.Level(PermissionLevel.GAMEMASTERS)))
            // /dumpnoise            (default radius 8)
            .executes(ctx -> execute(ctx, 8))
            // /dumpnoise <radius>
            .then(CommandManager.argument("radius", IntegerArgumentType.integer(1, 64))
                .executes(ctx -> execute(ctx,
                        IntegerArgumentType.getInteger(ctx, "radius"))))
        );
    }

    // ------------------------------------------------------------------
    // Main handler
    // ------------------------------------------------------------------

    private static int execute(CommandContext<ServerCommandSource> ctx, int radius) {
        ServerCommandSource source = ctx.getSource();
        ServerWorld world = source.getWorld();

        // Output directory: <run>/noise_dumps/
        Path outDir = Path.of("noise_dumps");
        try {
            Files.createDirectories(outDir);
        } catch (IOException e) {
            source.sendError(Text.literal("[NoiseDumper] Cannot create output dir: " + e.getMessage()));
            return 0;
        }

        // Attempt to get NoiseConfig for router6 extraction (best-effort — may be null).
        // Heightmaps and biomes are always dumped regardless.
        NoiseConfig noiseConfig = tryGetNoiseConfig(world);
        if (noiseConfig == null) {
            HelloTerrainMod.LOGGER.warn(
                    "[NoiseDumper] NoiseConfig unavailable — router6 will be omitted from dumps."
                    + " Heightmaps and biomes will still be captured.");
        }

        long seed = world.getSeed();
        BiomeAccess biomeAccess = world.getBiomeAccess();

        // Find player origin chunk (or fallback to 0,0)
        BlockPos origin;
        try {
            origin = BlockPos.ofFloored(source.getPosition());
        } catch (UnsupportedOperationException e) {
            origin = BlockPos.ORIGIN;
        }
        int centerCx = origin.getX() >> 4;
        int centerCz = origin.getZ() >> 4;

        source.sendFeedback(
                () -> Text.literal(String.format(
                        "[NoiseDumper] Dumping %d×%d chunks centred (%d,%d) → %s",
                        (2 * radius + 1), (2 * radius + 1), centerCx, centerCz, outDir.toAbsolutePath())),
                false);

        // Worker thread — chunk loading can be slow
        int[] counts = {0, 0};  // [dumped, skipped]
        Thread worker = new Thread(() -> {
            for (int dx = -radius; dx <= radius; dx++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    int cx = centerCx + dx;
                    int cz = centerCz + dz;

                    Chunk chunk = world.getChunkManager()
                            .getChunk(cx, cz, ChunkStatus.FULL, false);
                    if (chunk == null) {
                        counts[1]++;
                        continue;
                    }

                    try {
                        dumpChunk(chunk, noiseConfig, biomeAccess, seed, outDir);
                        counts[0]++;
                    } catch (Exception e) {
                        LOG.warning("[NoiseDumper] Failed chunk (" + cx + "," + cz + "): " + e);
                        counts[1]++;
                    }
                }
            }

            // Report back on main thread
            int dumped = counts[0];
            int skipped = counts[1];
            source.sendFeedback(
                    () -> Text.literal(String.format(
                            "[NoiseDumper] Done. Dumped %d chunks, skipped %d unloaded.",
                            dumped, skipped)),
                    false);
        }, "NoiseDumper-Worker");
        worker.setDaemon(true);
        worker.start();

        return 1;
    }

    // ------------------------------------------------------------------
    // Per-chunk dump
    // ------------------------------------------------------------------

    /**
     * Dump a single chunk's noise signals to JSON.
     *
     * <p>Always written: heightmap_surface, heightmap_ocean_floor, biomes4.
     * <p>Written when noiseConfig is non-null: router6 (requires NoiseTap).
     * <p>When router6 is absent the Python pipeline falls back to
     * {@code approximate_router6_from_biome} automatically.
     */
    static void dumpChunk(Chunk chunk,
                          NoiseConfig noiseConfig,
                          BiomeAccess biomeAccess,
                          long seed,
                          Path outDir) throws IOException {
        ChunkPos cp = chunk.getPos();
        String filename = String.format("chunk_%d_%d.json", cp.x, cp.z);
        Path file = outDir.resolve(filename);

        StringBuilder sb = new StringBuilder(8192);
        sb.append("{\n");
        sb.append("  \"chunk_x\": ").append(cp.x).append(",\n");
        sb.append("  \"chunk_z\": ").append(cp.z).append(",\n");
        sb.append("  \"seed\": ").append(seed).append(",\n");
        sb.append("  \"router6_available\": ").append(noiseConfig != null).append(",\n");

        // Router6 — only when NoiseConfig is available
        if (noiseConfig != null) {
            NoiseTap tap = NoiseTap.bind(chunk, noiseConfig, biomeAccess, seed);
            NoiseTap.Cache cache = tap.captureAll(CORE_FIELDS, HEIGHTMAPS);

            String[] fieldNames = {
                "temperature", "vegetation", "continents", "erosion", "depth", "ridges"
            };
            RouterField[] fields = {
                RouterField.TEMPERATURE, RouterField.VEGETATION, RouterField.CONTINENTS,
                RouterField.EROSION, RouterField.DEPTH, RouterField.RIDGES
            };
            sb.append("  \"router6\": {\n");
            for (int fi = 0; fi < 6; fi++) {
                float[][][] data = cache.getRouterField(fields[fi]);
                sb.append("    \"").append(fieldNames[fi]).append("\": [");
                for (int x = 0; x < 16; x++) {
                    for (int z = 0; z < 16; z++) {
                        for (int y = 0; y < 16; y++) {
                            if (x > 0 || z > 0 || y > 0) sb.append(',');
                            sb.append(data[x][z][y]);
                        }
                    }
                }
                sb.append(fi < 5 ? "],\n" : "]\n");
            }
            sb.append("  },\n");

            // Heightmaps from NoiseTap cache
            short[][] surfaceHm = cache.getHeightmap(Heightmap.Type.WORLD_SURFACE_WG);
            sb.append("  \"heightmap_surface\": [");
            appendShortGrid(sb, surfaceHm);
            sb.append("],\n");

            short[][] oceanHm = cache.getHeightmap(Heightmap.Type.OCEAN_FLOOR_WG);
            sb.append("  \"heightmap_ocean_floor\": [");
            appendShortGrid(sb, oceanHm);
            sb.append("],\n");

            // Biomes from NoiseTap cache
            sb.append("  \"biomes\": [");
            int[][][] biomes = cache.biomes4();
            for (int bx = 0; bx < 4; bx++) {
                for (int bz = 0; bz < 4; bz++) {
                    for (int by = 0; by < 4; by++) {
                        if (bx > 0 || bz > 0 || by > 0) sb.append(',');
                        sb.append(biomes[bx][bz][by]);
                    }
                }
            }
            sb.append("]\n");
        } else {
            // Heightmaps directly from Chunk API (always available)
            var surfaceHm = chunk.getHeightmap(Heightmap.Type.WORLD_SURFACE_WG);
            sb.append("  \"heightmap_surface\": [");
            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    if (x > 0 || z > 0) sb.append(',');
                    sb.append(surfaceHm.get(x, z));
                }
            }
            sb.append("],\n");

            var oceanHm = chunk.getHeightmap(Heightmap.Type.OCEAN_FLOOR_WG);
            sb.append("  \"heightmap_ocean_floor\": [");
            for (int x = 0; x < 16; x++) {
                for (int z = 0; z < 16; z++) {
                    if (x > 0 || z > 0) sb.append(',');
                    sb.append(oceanHm.get(x, z));
                }
            }
            sb.append("],\n");

            // Biomes at 4×4×4 lattice from chunk directly
            sb.append("  \"biomes\": [");
            boolean first = true;
            for (int bx = 0; bx < 4; bx++) {
                for (int bz = 0; bz < 4; bz++) {
                    for (int by = 0; by < 4; by++) {
                        if (!first) sb.append(',');
                        first = false;
                        RegistryEntry<Biome> b = chunk.getBiomeForNoiseGen(bx, by, bz);
                        sb.append(Math.abs(b.hashCode()) % 256);
                    }
                }
            }
            sb.append("]\n");
        }

        sb.append("}\n");
        Files.writeString(file, sb.toString());
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /**
     * Attempt to extract a {@link NoiseConfig} from the world's chunk generator.
     *
     * <p>In MC 1.21.x, {@code NoiseConfig} is created during world loading and
     * passed transiently through the chunk-generation pipeline — it is not exposed
     * via a public getter.  We try reflection as a best-effort approach; returning
     * {@code null} is graceful (the dump falls back to heightmaps + biomes only).
     */
    static NoiseConfig tryGetNoiseConfig(ServerWorld world) {
        try {
            ChunkGenerator gen = world.getChunkManager().getChunkGenerator();
            if (!(gen instanceof NoiseChunkGenerator)) return null;

            // Walk all declared fields looking for a cached NoiseConfig instance
            for (Class<?> cls = gen.getClass(); cls != null && cls != Object.class; cls = cls.getSuperclass()) {
                for (Field f : cls.getDeclaredFields()) {
                    if (NoiseConfig.class.isAssignableFrom(f.getType())) {
                        f.setAccessible(true);
                        NoiseConfig nc = (NoiseConfig) f.get(gen);
                        if (nc != null) return nc;
                    }
                }
            }
        } catch (Exception e) {
            HelloTerrainMod.LOGGER.warn("[NoiseDumper] Could not access NoiseConfig via reflection: {}",
                    e.getMessage());
        }
        return null;
    }

    private static void appendShortGrid(StringBuilder sb, short[][] grid) {
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (x > 0 || z > 0) sb.append(',');
                sb.append(grid[x][z]);
            }
        }
    }
}
