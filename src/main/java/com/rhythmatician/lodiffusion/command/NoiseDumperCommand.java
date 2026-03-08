package com.rhythmatician.lodiffusion.command;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.IntegerArgumentType;
import com.mojang.brigadier.context.CommandContext;
import com.rhythmatician.lodiffusion.voxy.WorldNoiseAccess;

import net.minecraft.command.permission.Permission;
import net.minecraft.command.permission.PermissionLevel;
import net.minecraft.server.command.CommandManager;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.server.world.ServerWorld;
import net.minecraft.text.Text;
import net.minecraft.util.math.BlockPos;
import net.minecraft.world.Heightmap;

/**
 * Server command {@code /dumpnoise <radius>} that extracts vanilla noise
 * signals using {@link WorldNoiseAccess} and serialises them to JSON files
 * under {@code run/noise_dumps/}.
 *
 * <p><b>No loaded chunks required.</b> All data is computed directly from
 * the {@link net.minecraft.world.gen.chunk.ChunkGenerator} and
 * {@link net.minecraft.world.gen.noise.NoiseConfig} — pure math, no world
 * state needed. This means noise can be dumped for <em>any</em> coordinate,
 * even if no player has ever visited the area.
 *
 * <p>Each dump file contains:
 * <ul>
 *   <li>{@code heightmap_surface} — 16×16 WORLD_SURFACE_WG heights (x-major)</li>
 *   <li>{@code heightmap_ocean_floor} — 16×16 OCEAN_FLOOR_WG heights (x-major)</li>
 *   <li>{@code router6} — 6 density-router channels × 256 surface samples
 *       (temperature, vegetation, continents, erosion, depth, ridges)</li>
 *   <li>{@code biomes} — 16×16 biome indices at block resolution (x-major)</li>
 *   <li>{@code seed}, {@code chunk_x}, {@code chunk_z}</li>
 * </ul>
 *
 * <p>Usage: {@code /dumpnoise [radius]}  (default radius = 8 chunks)
 *
 * <p>The output JSON can be consumed by the Python training pipeline through
 * {@code scripts/add_column_heights.py} when the
 * {@code --noise-dump-dir} option is supplied.
 */
public final class NoiseDumperCommand {

    private static final Logger LOG = Logger.getLogger(NoiseDumperCommand.class.getName());

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

        // Create WorldNoiseAccess — chunk-free noise pipeline.
        // If this fails, we cannot proceed (no fallback to chunk-based sampling).
        WorldNoiseAccess noise = WorldNoiseAccess.tryCreate(world);
        if (noise == null) {
            source.sendError(Text.literal(
                    "[NoiseDumper] Failed to initialise noise pipeline. "
                    + "NoiseConfig unavailable — is this a vanilla overworld?"));
            return 0;
        }

        long seed = world.getSeed();

        // Find player origin chunk (or fallback to 0,0)
        BlockPos origin;
        try {
            origin = BlockPos.ofFloored(source.getPosition());
        } catch (UnsupportedOperationException e) {
            origin = BlockPos.ORIGIN;
        }
        int centerCx = origin.getX() >> 4;
        int centerCz = origin.getZ() >> 4;

        int totalChunks = (2 * radius + 1) * (2 * radius + 1);
        source.sendFeedback(
                () -> Text.literal(String.format(
                        "[NoiseDumper] Dumping %d chunks (%d×%d) centred (%d,%d) → %s",
                        totalChunks, 2 * radius + 1, 2 * radius + 1,
                        centerCx, centerCz, outDir.toAbsolutePath())),
                false);

        // Worker thread — sampling many chunks can take a while
        Thread worker = new Thread(() -> {
            int dumped = 0;
            int failed = 0;

            for (int dx = -radius; dx <= radius; dx++) {
                for (int dz = -radius; dz <= radius; dz++) {
                    int cx = centerCx + dx;
                    int cz = centerCz + dz;

                    try {
                        dumpChunkNoise(noise, cx, cz, seed, outDir);
                        dumped++;
                    } catch (Exception e) {
                        LOG.warning("[NoiseDumper] Failed chunk (" + cx + "," + cz + "): " + e);
                        failed++;
                    }
                }

                // Progress feedback every row
                final int row = dx + radius + 1;
                final int rows = 2 * radius + 1;
                source.sendFeedback(
                        () -> Text.literal(String.format(
                                "[NoiseDumper] Progress: row %d/%d", row, rows)),
                        false);
            }

            final int d = dumped;
            final int f = failed;
            source.sendFeedback(
                    () -> Text.literal(String.format(
                            "[NoiseDumper] Done. Dumped %d chunks, %d failed.", d, f)),
                    false);
        }, "NoiseDumper-Worker");
        worker.setDaemon(true);
        worker.start();

        return 1;
    }

    // ------------------------------------------------------------------
    // Per-chunk dump (chunk-free)
    // ------------------------------------------------------------------

    /**
     * Dump noise signals for a single chunk position to JSON.
     *
     * <p>All data is computed via {@link WorldNoiseAccess} — no loaded chunk
     * or world state is required.
     *
     * @param noise  the noise access (provides heightmaps, router6, biomes)
     * @param cx     chunk X coordinate
     * @param cz     chunk Z coordinate
     * @param seed   world seed
     * @param outDir output directory
     */
    static void dumpChunkNoise(WorldNoiseAccess noise,
                               int cx, int cz, long seed,
                               Path outDir) throws IOException {
        String filename = String.format("chunk_%d_%d.json", cx, cz);
        Path file = outDir.resolve(filename);

        // Sample heightmaps (chunk-free via ChunkGenerator.getHeight())
        float[][] surfaceHm = noise.sampleHeightmap(cx, cz,
                Heightmap.Type.WORLD_SURFACE_WG);
        float[][] oceanHm = noise.sampleHeightmap(cx, cz,
                Heightmap.Type.OCEAN_FLOOR_WG);

        // Sample router6 at surface level (chunk-free via DensityFunction.sample())
        float[][] router6 = noise.sampleRouter6(cx, cz, surfaceHm);

        // Sample biomes at surface level (chunk-free via BiomeSource.getBiome())
        int[][] biomes = noise.sampleBiomes(cx, cz, surfaceHm);

        // Build JSON
        StringBuilder sb = new StringBuilder(8192);
        sb.append("{\n");
        sb.append("  \"chunk_x\": ").append(cx).append(",\n");
        sb.append("  \"chunk_z\": ").append(cz).append(",\n");
        sb.append("  \"seed\": ").append(seed).append(",\n");
        sb.append("  \"router6_available\": true,\n");

        // Heightmaps — flat 256 values, x-major (x outer, z inner)
        sb.append("  \"heightmap_surface\": [");
        appendFloatGrid(sb, surfaceHm);
        sb.append("],\n");

        sb.append("  \"heightmap_ocean_floor\": [");
        appendFloatGrid(sb, oceanHm);
        sb.append("],\n");

        // Router6 — 6 channels × 256 values each (x-major within channel)
        String[] fieldNames = {
            "temperature", "vegetation", "continents", "erosion", "depth", "ridges"
        };
        sb.append("  \"router6\": {\n");
        for (int fi = 0; fi < 6; fi++) {
            sb.append("    \"").append(fieldNames[fi]).append("\": [");
            float[] channel = router6[fi];
            for (int i = 0; i < channel.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(channel[i]);
            }
            sb.append(fi < 5 ? "],\n" : "]\n");
        }
        sb.append("  },\n");

        // Biomes — flat 256 values, x-major (block resolution)
        sb.append("  \"biomes\": [");
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (x > 0 || z > 0) sb.append(',');
                sb.append(biomes[x][z]);
            }
        }
        sb.append("]\n");

        sb.append("}\n");
        Files.writeString(file, sb.toString());
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    private static void appendFloatGrid(StringBuilder sb, float[][] grid) {
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (x > 0 || z > 0) sb.append(',');
                // Cast to int — heightmaps are whole-block Y values
                sb.append((int) grid[x][z]);
            }
        }
    }
}
