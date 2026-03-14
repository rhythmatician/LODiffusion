package com.rhythmatician.lodiffusion.command;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.Future;
import java.util.concurrent.atomic.AtomicInteger;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

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
import net.minecraft.world.gen.noise.NoiseRouter;

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
 *   <li>{@code biome_names} — 16×16 biome registry key names at block resolution
 *       (e.g. "minecraft:plains"), x-major</li>
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

    private static final Logger LOG = LoggerFactory.getLogger(NoiseDumperCommand.class);

    private NoiseDumperCommand() {}

    /**
     * Register {@code /dumpnoise [radius]} and {@code /dumpnoise parity <cx> <cz>}
     * with the Brigadier dispatcher.
     * Should be called from {@link com.rhythmatician.lodiffusion.HelloTerrainMod}.
     */
    public static void register(CommandDispatcher<ServerCommandSource> dispatcher) {
        dispatcher.register(CommandManager.literal("dumpnoise")
            .requires(src -> src.getPermissions().hasPermission(
                    new Permission.Level(PermissionLevel.GAMEMASTERS)))
            // /dumpnoise            (default radius 8)
            .executes(ctx -> execute(ctx, 8))
            // /dumpnoise <radius>
            .then(CommandManager.argument("radius", IntegerArgumentType.integer(1, 512))
                .executes(ctx -> execute(ctx,
                        IntegerArgumentType.getInteger(ctx, "radius"))))
            // /dumpnoise parity <cx> <cz>
            // Writes block-resolution Java vanilla density to parity_reports/java_chunk_<cx>_<cz>.json
            // Compare against parity_reports/gpu_chunk_0_0.json (auto-written at world load)
            // then run:  python tools/validate_shader_parity.py
            .then(CommandManager.literal("parity")
                .then(CommandManager.argument("cx", IntegerArgumentType.integer(-512, 512))
                    .then(CommandManager.argument("cz", IntegerArgumentType.integer(-512, 512))
                        .executes(ctx -> executeParity(ctx,
                                IntegerArgumentType.getInteger(ctx, "cx"),
                                IntegerArgumentType.getInteger(ctx, "cz"))))))
        );
    }

    // ------------------------------------------------------------------
    // Parity handler (/dumpnoise parity <cx> <cz>)
    // ------------------------------------------------------------------

    /**
     * WS-1.3: Dump block-resolution Java vanilla final_density for one chunk.
     *
     * <p>Writes {@code run/parity_reports/java_chunk_<cx>_<cz>.json} with a flat
     * {@code density} array of 98,304 floats indexed {@code [lx + 16*lz] * 384 + (by + 64)}.
     * Compare against {@code parity_reports/gpu_chunk_0_0.json} (auto-written at world load)
     * using {@code python tools/validate_shader_parity.py}.
     */
    private static int executeParity(CommandContext<ServerCommandSource> ctx,
                                     int cx, int cz) {
        ServerCommandSource source = ctx.getSource();
        ServerWorld world = source.getWorld();

        WorldNoiseAccess noise = WorldNoiseAccess.tryCreate(world);
        if (noise == null) {
            source.sendError(Text.literal("[NoiseDumper] parity: noise pipeline unavailable"));
            return 0;
        }

        source.sendFeedback(
                () -> Text.literal(String.format(
                        "[NoiseDumper] parity: sampling Java density for chunk (%d,%d) — 98,304 positions...",
                        cx, cz)),
                false);

        long t0 = System.currentTimeMillis();
        NoiseRouter router = noise.getNoiseRouter();
        float[] javaDensity = noise.sampleFinalDensityBlockRes(router, cx, cz);
        long samplingMs = System.currentTimeMillis() - t0;

        // Compute a quick surface-level mean to verify non-trivial output
        float sum = 0f;
        for (int col = 0; col < 256; col++) {
            // Y=63 (sea level): index = col*384 + (63+64) = col*384 + 127
            sum += javaDensity[col * 384 + 127];
        }
        float meanAtSeaLevel = sum / 256f;

        source.sendFeedback(
                () -> Text.literal(String.format(
                        "[NoiseDumper] parity: sampled 98,304 values in %dms; mean density @ Y=63: %.5f",
                        samplingMs, meanAtSeaLevel)),
                false);

        // Write JSON
        try {
            Path dir = Path.of("parity_reports");
            Files.createDirectories(dir);
            Path out = dir.resolve(String.format("java_chunk_%d_%d.json", cx, cz));

            StringBuilder sb = new StringBuilder(2 * 1024 * 1024);
            sb.append("{\n");
            sb.append("  \"chunk_x\": ").append(cx).append(",\n");
            sb.append("  \"chunk_z\": ").append(cz).append(",\n");
            sb.append("  \"source\": \"java\",\n");
            sb.append("  \"y_min\": -64,\n");
            sb.append("  \"y_levels\": 384,\n");
            sb.append("  \"note\": \"density[lx + 16*lz][by - y_min]\",\n");
            sb.append("  \"density\": [");
            for (int i = 0; i < javaDensity.length; i++) {
                if (i > 0) sb.append(',');
                sb.append(String.format("%.6g", javaDensity[i]));
            }
            sb.append("]\n}\n");

            Files.writeString(out, sb);
            final long sizeKB = out.toFile().length() / 1024;
            source.sendFeedback(
                    () -> Text.literal(String.format(
                            "[NoiseDumper] parity: wrote %s (%dKB) — now run: python tools/validate_shader_parity.py",
                            out.toAbsolutePath(), sizeKB)),
                    false);
            return 1;
        } catch (IOException e) {
            source.sendError(Text.literal("[NoiseDumper] parity: write failed — " + e.getMessage()));
            return 0;
        }
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

        // Parallel worker pool — limited to avoid starving the server tick loop.
        // ChunkStatus.NOISE generation runs on the server's own worker executor
        // internally, so each task here just blocks waiting for that result.
        // 4 threads is a good balance: enough to keep the pipeline full without
        // hammering the CPU the way 31 threads did.
        int threadCount = 4;
        source.sendFeedback(
                () -> Text.literal(String.format(
                        "[NoiseDumper] Using %d worker threads", threadCount)),
                false);

        ExecutorService pool = Executors.newFixedThreadPool(threadCount, r -> {
            Thread t = new Thread(r, "NoiseDumper-Worker");
            t.setDaemon(true);
            return t;
        });

        AtomicInteger dumped = new AtomicInteger();
        AtomicInteger failed = new AtomicInteger();
        long startTime = System.currentTimeMillis();
        List<Future<?>> futures = new ArrayList<>(totalChunks);

        for (int dx = -radius; dx <= radius; dx++) {
            for (int dz = -radius; dz <= radius; dz++) {
                final int cx = centerCx + dx;
                final int cz = centerCz + dz;
                futures.add(pool.submit(() -> {
                    try {
                        dumpChunkNoise(noise, cx, cz, seed, outDir);
                        dumped.incrementAndGet();
                    } catch (Exception e) {
                        LOG.warn("[NoiseDumper] Failed chunk (" + cx + "," + cz + "): " + e);
                        failed.incrementAndGet();
                    }

                    // Throttled progress: every 100 chunks or on the last one
                    int done = dumped.get() + failed.get();
                    if (done % 100 == 0 || done == totalChunks) {
                        double elapsed = (System.currentTimeMillis() - startTime) / 1000.0;
                        double rate = elapsed > 0 ? done / elapsed : 0;
                        double eta = rate > 0 ? (totalChunks - done) / rate : 0;
                        source.sendFeedback(
                                () -> Text.literal(String.format(
                                        "[NoiseDumper] %d/%d (%.1f/s, ETA %.0fs)",
                                        done, totalChunks, rate, eta)),
                                false);
                    }
                }));
            }
        }

        // Coordinator thread waits for all futures, then shuts down the pool
        Thread coordinator = new Thread(() -> {
            for (Future<?> f : futures) {
                try {
                    f.get();
                } catch (Exception e) {
                    LOG.warn("[NoiseDumper] Future error: " + e);
                }
            }
            pool.shutdown();
            double totalSec = (System.currentTimeMillis() - startTime) / 1000.0;
            final int d = dumped.get();
            final int fl = failed.get();
            source.sendFeedback(
                    () -> Text.literal(String.format(
                            "[NoiseDumper] Done. %d dumped, %d failed in %.1fs (%.1f chunks/s)",
                            d, fl, totalSec, d / totalSec)),
                    false);
        }, "NoiseDumper-Coordinator");
        coordinator.setDaemon(true);
        coordinator.start();

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
     * @param noise  the noise access (provides heightmaps, biomes)
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

        // Sample both heightmaps in one populateNoise() pass via ChunkStatus.NOISE.
        // This is ~64× cheaper than calling getHeight() 512 times (256 cols × 2 types)
        // because ChunkNoiseSampler reuses noise evaluations across the 4×4 cell grid.
        float[][][] heightmaps = noise.sampleBothHeightmaps(cx, cz);
        float[][] surfaceHm   = heightmaps[0];  // WORLD_SURFACE_WG
        float[][] oceanHm     = heightmaps[1];  // OCEAN_FLOOR_WG

        // Sample biomes at surface level (chunk-free via BiomeSource.getBiome())
        String[][] biomeNames = noise.sampleBiomeNames(cx, cz, surfaceHm);

        // ---- Raw NoiseRouter fields ----------------------------------------
        // Sea level Y=63 is a good reference for the 2D-ish climate fields.
        // These functions vary very little with Y except near terrain transitions.
        NoiseRouter router = noise.getNoiseRouter();
        final int seaLevel = 63;

        // 2D climate/shape fields (4×4 cell grid, sampled at sea level)
        float[][] continents  = noise.sampleRouterField2D(router.continents(),  cx, cz, seaLevel);
        float[][] erosion     = noise.sampleRouterField2D(router.erosion(),     cx, cz, seaLevel);
        float[][] ridges      = noise.sampleRouterField2D(router.ridges(),      cx, cz, seaLevel);
        float[][] temperature = noise.sampleRouterField2D(router.temperature(), cx, cz, seaLevel);
        float[][] vegetation  = noise.sampleRouterField2D(router.vegetation(),  cx, cz, seaLevel);
        float[][] depth       = noise.sampleRouterField2D(router.depth(),       cx, cz, seaLevel);

        // 3D density field (4×48×4 cell grid, full overworld Y range -64..320)
        float[][][] finalDensity = noise.sampleRouterField3D(router.finalDensity(), cx, cz);
        // -------------------------------------------------------------------

        // Build JSON
        StringBuilder sb = new StringBuilder(8192);
        sb.append("{\n");
        sb.append("  \"chunk_x\": ").append(cx).append(",\n");
        sb.append("  \"chunk_z\": ").append(cz).append(",\n");
        sb.append("  \"seed\": ").append(seed).append(",\n");

        // Heightmaps — flat 256 values, x-major (x outer, z inner)
        sb.append("  \"heightmap_surface\": [");
        appendFloatGrid(sb, surfaceHm);
        sb.append("],\n");

        sb.append("  \"heightmap_ocean_floor\": [");
        appendFloatGrid(sb, oceanHm);
        sb.append("],\n");

        // Biome names — flat 256 strings, x-major (block resolution)
        sb.append("  \"biome_names\": [");
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (x > 0 || z > 0) sb.append(',');
                sb.append('"').append(biomeNames[x][z]).append('"');
            }
        }
        sb.append("],\n");

        // 2D router fields — flat 16 values each (4×4), cx-outer / cz-inner
        sb.append("  \"continents\": [");
        appendCell2D(sb, continents);
        sb.append("],\n");

        sb.append("  \"erosion\": [");
        appendCell2D(sb, erosion);
        sb.append("],\n");

        sb.append("  \"ridges\": [");
        appendCell2D(sb, ridges);
        sb.append("],\n");

        sb.append("  \"temperature\": [");
        appendCell2D(sb, temperature);
        sb.append("],\n");

        sb.append("  \"vegetation\": [");
        appendCell2D(sb, vegetation);
        sb.append("],\n");

        sb.append("  \"depth\": [");
        appendCell2D(sb, depth);
        sb.append("],\n");

        // 3D final_density — flat 768 values (4×48×4), cx-outer / cy-middle / cz-inner
        sb.append("  \"final_density\": [");
        appendCell3D(sb, finalDensity);
        sb.append("]\n");

        sb.append("}\n");
        Files.writeString(file, sb.toString());
    }

    // ------------------------------------------------------------------
    // Helpers
    // ------------------------------------------------------------------

    /** Append a flat 16×16 heightmap as integers (x-major). */
    private static void appendFloatGrid(StringBuilder sb, float[][] grid) {
        for (int x = 0; x < 16; x++) {
            for (int z = 0; z < 16; z++) {
                if (x > 0 || z > 0) sb.append(',');
                // Cast to int — heightmaps are whole-block Y values
                sb.append((int) grid[x][z]);
            }
        }
    }

    /**
     * Append a 4×4 float cell grid as a flat JSON array (cx-outer, cz-inner).
     * Values are written as floating-point with 6 significant digits.
     */
    private static void appendCell2D(StringBuilder sb, float[][] grid) {
        boolean first = true;
        for (int cx = 0; cx < 4; cx++) {
            for (int cz = 0; cz < 4; cz++) {
                if (!first) sb.append(',');
                first = false;
                sb.append(String.format("%.6g", grid[cx][cz]));
            }
        }
    }

    /**
     * Append a 4×48×4 float cell grid as a flat JSON array (cx-outer, cy-middle, cz-inner).
     */
    private static void appendCell3D(StringBuilder sb, float[][][] grid) {
        boolean first = true;
        for (int cx = 0; cx < 4; cx++) {
            for (int cy = 0; cy < 48; cy++) {
                for (int cz = 0; cz < 4; cz++) {
                    if (!first) sb.append(',');
                    first = false;
                    sb.append(String.format("%.6g", grid[cx][cy][cz]));
                }
            }
        }
    }
}
