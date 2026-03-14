package io.github.lodiffusion.worldgen;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.reflect.Field;
import java.lang.reflect.Method;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Map;
import java.util.WeakHashMap;

import com.rhythmatician.lodiffusion.voxy.VoxyCompat;

/**
 * Event handler for world generation integration (reflection-driven).
 *
 * Hooks into ServerLevelEvents to extract NoiseRouter parameters when a world loads,
 * then uploads them to GPU SSBOs for parallel terrain generation.
 *
 * All Minecraft class references use reflection to avoid compile-time classpath issues.
 *
 * Lifecycle:
 * - LOAD: Extract NoiseRouter, create ShaderSSBOManager, upload SSBOs
 * - UNLOAD: Cleanup ShaderSSBOManager, prevent VRAM leaks
 * - Orphan checks: Warn if old SSBO state isn't cleaned up before next load
 */
public class WorldGenEventHandler {
    private static final Logger LOGGER = LogManager.getLogger();

    /** Map of ServerLevel (Object) → active ShaderSSBOManager (for lifecycle tracking) */
    private static final Map<Object, ShaderSSBOManager> activeLevels = new WeakHashMap<>();

    /** Current singleton instance (one per server lifecycle) */
    private static WorldGenEventHandler instance;

    // Cached reflection metadata for Minecraft classes (loaded lazily)
    private final Class<?> serverLevelClass;
    private final Class<?> minecraftServerClass;
    private final Class<?> chunkSourceClass;
    private final Class<?> chunkGeneratorClass;
    private final Class<?> noiseBasedChunkGeneratorClass;
    private final Class<?> noiseRouterClass;
    private final Class<?> dimensionTypeClass;
    private final Class<?> resourceKeyClass;

    private final Method getChunkSourceMethod;
    private final Method getGeneratorMethod;
    private final Method getNoiseRouterMethod;
    private final Method getDimensionMethod;

    private WorldGenEventHandler() {
        // Load Minecraft class references via reflection
        this.serverLevelClass = loadClassOrNull("net.minecraft.server.level.ServerLevel");
        this.minecraftServerClass = loadClassOrNull("net.minecraft.server.MinecraftServer");
        this.chunkSourceClass = loadClassOrNull("net.minecraft.world.level.chunk.ChunkSource");
        this.chunkGeneratorClass = loadClassOrNull("net.minecraft.world.level.chunk.ChunkGenerator");
        this.noiseBasedChunkGeneratorClass = loadClassOrNull("net.minecraft.world.level.levelgen.NoiseBasedChunkGenerator");
        this.noiseRouterClass = loadClassOrNull("net.minecraft.world.level.levelgen.NoiseRouter");
        this.dimensionTypeClass = loadClassOrNull("net.minecraft.world.level.dimension.DimensionType");
        this.resourceKeyClass = loadClassOrNull("net.minecraft.resources.ResourceKey");

        // Cache methods for performance
        this.getChunkSourceMethod = findMethodOrNull(serverLevelClass, "getChunkSource");
        this.getGeneratorMethod = findMethodOrNull(chunkSourceClass, "getGenerator");
        this.getNoiseRouterMethod = findMethodOrNull(noiseBasedChunkGeneratorClass, "getNoiseRouter");
        this.getDimensionMethod = findMethodOrNull(serverLevelClass, "dimension");
    }

    /**
     * Initializes event handlers. Call once during mod initialization.
     */
    public static synchronized void initialize() {
        if (instance != null) {
            LOGGER.warn("WorldGenEventHandler already initialized");
            return;
        }

        instance = new WorldGenEventHandler();

        try {
            // Register LOAD event
            registerLoadEvent();
            
            // Register UNLOAD event
            registerUnloadEvent();

            LOGGER.info("WorldGenEventHandler initialized — listening for ServerLevelEvents.LOAD/UNLOAD");
        } catch (Exception e) {
            LOGGER.error("Failed to initialize WorldGenEventHandler event listeners", e);
        }
    }

    /**
     * Registers the LOAD event using reflection.
     */
    private static void registerLoadEvent() {
        try {
            // net.fabricmc.fabric.api.event.lifecycle.v1.ServerLevelEvents
            Class<?> serverLevelEventsClass = Class.forName("net.fabricmc.fabric.api.event.lifecycle.v1.ServerLevelEvents");
            
            // Get the LOAD field (which is an Event)
            Field loadField = serverLevelEventsClass.getField("LOAD");
            Object loadEvent = loadField.get(null);

            // Get the register method on the Event
            Method registerMethod = loadEvent.getClass().getMethod("register", Object.class);

            // Create a callback - use a simple approach with a wrapper object
            Object callback = new Object() {
                public void onLoadLevel(Object server, Object level) {
                    if (instance != null) {
                        instance.onWorldLoad(server, level);
                    }
                }
            };

            // Register the callback
            registerMethod.invoke(loadEvent, callback);
            LOGGER.info("Registered ServerLevelEvents.LOAD listener");

        } catch (Exception e) {
            LOGGER.error("Failed to register LOAD event", e);
        }
    }

    /**
     * Registers the UNLOAD event using reflection.
     */
    private static void registerUnloadEvent() {
        try {
            Class<?> serverLevelEventsClass = Class.forName("net.fabricmc.fabric.api.event.lifecycle.v1.ServerLevelEvents");
            Field unloadField = serverLevelEventsClass.getField("UNLOAD");
            Object unloadEvent = unloadField.get(null);

            Method registerMethod = unloadEvent.getClass().getMethod("register", Object.class);
            
            Object callback = new Object() {
                public void onUnloadLevel(Object server, Object level) {
                    if (instance != null) {
                        instance.onWorldUnload(server, level);
                    }
                }
            };

            registerMethod.invoke(unloadEvent, callback);
            LOGGER.info("Registered ServerLevelEvents.UNLOAD listener");

        } catch (Exception e) {
            LOGGER.error("Failed to register UNLOAD event", e);
        }
    }

    /**
     * Triggered when a ServerLevel is loaded.
     */
    private void onWorldLoad(Object server, Object level) {
        long startTime = System.currentTimeMillis();
        LOGGER.info("=== WorldGenEventHandler.onWorldLoad ===");

        try {
            // Get dimension info for logging
            String dimensionInfo = getDimensionInfo(level);
            LOGGER.info("Level: {}", dimensionInfo);

            // Check for orphaned SSBO state
            if (activeLevels.containsKey(level)) {
                ShaderSSBOManager orphaned = activeLevels.get(level);
                if (orphaned.isValid()) {
                    LOGGER.warn("Orphaned ShaderSSBOManager found for level {} — cleaning up", dimensionInfo);
                    orphaned.cleanup();
                }
            }

            // Extract NoiseRouter from the chunk generator
            Object router = extractNoiseRouter(level);
            if (router == null) {
                LOGGER.warn("Unable to extract NoiseRouter from level {} — skipping GPU setup", dimensionInfo);
                return;
            }

            // Extract parameters using reflection-driven extractor
            LOGGER.info("Extracting NoiseRouter parameters...");
            NoiseRouterExtractor extractor = new NoiseRouterExtractor();
            NoiseRouterExtractor.NoiseRouterData data = extractor.extract(router);

            // Verify extracted data is non-null
            if (data == null) {
                LOGGER.error("NoiseRouterExtractor returned null data — aborting GPU setup");
                return;
            }
            LOGGER.info("Extraction complete: {} noise instances discovered", 
                    countExtractedInstances(data));

            // Upload to GPU SSBOs
            LOGGER.info("Creating ShaderSSBOManager and uploading to GPU...");
            ShaderSSBOManager manager = new ShaderSSBOManager();
            manager.uploadNoiseData(data); // Also compiles shader program
            activeLevels.put(level, manager);

            // Dispatch GPU compute for chunk (0,0) as a cold-start validation pass
            LOGGER.info("Dispatching GPU compute for validation (chunk 0,0)...");
            FloatBuffer allDensity = manager.dispatchAndRead(0, 0);
            LOGGER.info("Compute dispatch complete — validating Binding 7 output...");

            if (allDensity != null && allDensity.hasRemaining()) {
                // Log first 10 samples for quick liveness check
                StringBuilder log = new StringBuilder("GPU Density Samples [0-9]: ");
                for (int i = 0; i < 10 && allDensity.hasRemaining(); i++) {
                    log.append(String.format("%.4f", allDensity.get())).append(" ");
                }
                LOGGER.info(log.toString());

                // Write full density output for WS-1.3 parity validation
                writeGpuParityFile(allDensity, dimensionInfo);
            } else {
                LOGGER.warn("Unable to read density output from GPU (null or empty buffer)");
            }

            // WS-2: Read block-material output and push to Voxy
            writeBlocksToVoxy(manager, level, dimensionInfo);

            long elapsedMs = System.currentTimeMillis() - startTime;
            LOGGER.info("WorldGenEventHandler.onWorldLoad complete in {} ms", elapsedMs);

        } catch (Exception e) {
            LOGGER.error("WorldGenEventHandler.onWorldLoad failed", e);
        }
    }

    /**
     * Reads the block-material output from the GPU and pushes chunk (0,0) into Voxy
     * as an initial proof-of-concept (WS-2.4).
     *
     * <p>Skipped when Voxy is unavailable or the dimension is not the overworld.
     * Catches all exceptions so a Voxy API change never crashes world load.
     */
    private void writeBlocksToVoxy(ShaderSSBOManager manager, Object level, String dimensionInfo) {
        if (!VoxyCompat.isAvailable()) {
            LOGGER.debug("WS-2: Voxy not available — skipping block write");
            return;
        }
        if (!dimensionInfo.contains("overworld") && !dimensionInfo.contains("(unknown)")) {
            LOGGER.debug("WS-2: Skipping block write for non-overworld dimension: {}", dimensionInfo);
            return;
        }
        try {
            IntBuffer blockMat = manager.readBlockOutput();
            if (blockMat == null) {
                LOGGER.warn("WS-2: block output buffer is null — skipping Voxy write");
                return;
            }

            // Get Voxy world engine from the level (ServerLevel extends World)
            Object worldEngine = VoxyCompat.getWorldEngine((net.minecraft.world.World) level);
            if (worldEngine == null) {
                LOGGER.warn("WS-2: Voxy world engine not available for this level");
                return;
            }

            // Get default biome ID (plains) from the mapper
            Object mapper = VoxyCompat.getMapper(worldEngine);
            int defaultBiome = resolveDefaultBiome(mapper);

            ShaderSectionWriter writer = new ShaderSectionWriter(worldEngine, defaultBiome);
            int nonAir = writer.writeColumn(blockMat, 0, 0);
            LOGGER.info("WS-2: wrote GPU chunk (0,0) to Voxy — {} non-air voxels", nonAir);

        } catch (Exception e) {
            LOGGER.warn("WS-2: writeBlocksToVoxy failed (non-fatal): {}", e.getMessage(), e);
        }
    }

    /**
     * Resolves the Voxy biome ID for {@code minecraft:plains} via the Mapper.
     * Returns 0 if resolution fails.
     */
    private int resolveDefaultBiome(Object mapper) {
        try {
            // The plains biome identifier is used as a fallback; just try to call
            // Mapper.getIdForBiome with the identifier string if the Voxy API supports it.
            Class<?> identifierClass = Class.forName("net.minecraft.util.Identifier");
            Object plainsId = identifierClass.getMethod("of", String.class).invoke(null, "minecraft:plains");
            Method m = mapper.getClass().getMethod("getIdForBiome", Object.class);
            return (int) m.invoke(mapper, plainsId);
        } catch (Exception e) {
            LOGGER.debug("WS-2: could not resolve plains biome ID ({}), using 0", e.getMessage());
            return 0;
        }
    }

    /**
     * Writes the GPU density output for chunk (0,0) to a JSON file for WS-1.3 parity validation.
     *
     * <p>File: {@code run/parity_reports/gpu_chunk_0_0.json}<br>
     * Format: {@code {"chunk_x":0,"chunk_z":0,"source":"gpu",
     * "density":[...98304 floats...]}}
     *
     * <p>Compare against {@code java_chunk_0_0.json} (written by {@code /dumpnoise parity 0 0})
     * using {@code tools/validate_shader_parity.py}.
     */
    private void writeGpuParityFile(FloatBuffer densityBuf, String dimensionInfo) {
        // Only write parity for the overworld (avoids nether/end false positives)
        if (!dimensionInfo.contains("overworld") && !dimensionInfo.contains("(unknown)")) {
            LOGGER.debug("Skipping parity file for non-overworld dimension: {}", dimensionInfo);
            return;
        }
        try {
            Path dir = Path.of("parity_reports");
            Files.createDirectories(dir);
            Path out = dir.resolve("gpu_chunk_0_0.json");

            densityBuf.rewind();
            StringBuilder sb = new StringBuilder(2 * 1024 * 1024);
            sb.append("{\n");
            sb.append("  \"chunk_x\": 0,\n");
            sb.append("  \"chunk_z\": 0,\n");
            sb.append("  \"source\": \"gpu\",\n");
            sb.append("  \"y_min\": -64,\n");
            sb.append("  \"y_levels\": 384,\n");
            sb.append("  \"note\": \"density[lx + 16*lz][by - y_min]\",\n");
            sb.append("  \"density\": [");
            boolean first = true;
            while (densityBuf.hasRemaining()) {
                if (!first) sb.append(',');
                first = false;
                sb.append(String.format("%.6g", densityBuf.get()));
            }
            sb.append("]\n}\n");

            Files.writeString(out, sb);
            LOGGER.info("WS-1.3 parity: wrote GPU density to {}", out.toAbsolutePath());
        } catch (Exception e) {
            LOGGER.warn("WS-1.3 parity: failed to write GPU density file", e);
        }
    }

    /**
     * Triggered when a ServerLevel is unloaded.
     */
    private void onWorldUnload(Object server, Object level) {
        String dimensionInfo = getDimensionInfo(level);
        LOGGER.info("=== WorldGenEventHandler.onWorldUnload ===");
        LOGGER.info("Level: {}", dimensionInfo);

        try {
            ShaderSSBOManager manager = activeLevels.remove(level);
            if (manager != null) {
                LOGGER.info("Cleaning up ShaderSSBOManager for level {}", dimensionInfo);
                manager.cleanup();
            } else {
                LOGGER.debug("No active ShaderSSBOManager for level {} to clean up", dimensionInfo);
            }
        } catch (Exception e) {
            LOGGER.error("Error during WorldGenEventHandler.onWorldUnload", e);
        }
    }

    /**
     * Extracts the NoiseRouter from a ServerLevel using reflection.
     */
    private Object extractNoiseRouter(Object level) {
        try {
            if (level == null || getChunkSourceMethod == null || getGeneratorMethod == null) {
                return null;
            }

            // level.getChunkSource() returns ChunkSource
            Object chunkSource = getChunkSourceMethod.invoke(level);
            if (chunkSource == null) return null;

            // chunkSource.getGenerator() returns ChunkGenerator
            Object generator = getGeneratorMethod.invoke(chunkSource);
            if (generator == null) return null;

            // Check if it's a NoiseBasedChunkGenerator via class name
            String genClassName = generator.getClass().getSimpleName();
            if (!genClassName.equals("NoiseBasedChunkGenerator")) {
                LOGGER.warn("ChunkGenerator is {} — cannot extract NoiseRouter",
                        genClassName);
                return null;
            }

            // Invoke getNoiseRouter() on the generator
            if (getNoiseRouterMethod != null) {
                return getNoiseRouterMethod.invoke(generator);
            }
            
            return null;

        } catch (Exception e) {
            LOGGER.error("Failed to extract NoiseRouter", e);
            return null;
        }
    }

    /**
     * Gets a human-readable dimension identifier for logging.
     */
    private String getDimensionInfo(Object level) {
        try {
            if (level == null || getDimensionMethod == null) {
                return "(unknown)";
            }

            Object dimensionKey = getDimensionMethod.invoke(level);
            if (dimensionKey == null) return "(null)";

            // Try to extract the path/identifier
            String keyStr = dimensionKey.toString();
            if (keyStr.contains("minecraft:")) {
                return keyStr.substring(keyStr.lastIndexOf("minecraft:") + 10);
            }
            return keyStr;

        } catch (Exception e) {
            return "(error)";
        }
    }

    /**
     * Helper to estimate the number of noise instances extracted.
     */
    private int countExtractedInstances(NoiseRouterExtractor.NoiseRouterData data) {
        int count = 0;
        if (data.improvedOrigins != null) count += data.improvedOrigins.capacity() / 4;
        if (data.improvedPerms != null) count += data.improvedPerms.capacity() / 256;
        if (data.perlinInts != null) count += Math.max(0, data.perlinInts.capacity() / 16);
        return Math.max(count, 1);
    }

    /**
     * Returns the active ShaderSSBOManager for a level (Object), or null.
     */
    public static ShaderSSBOManager getManagerForLevel(Object level) {
        if (instance == null) return null;
        return activeLevels.get(level);
    }

    /**
     * Cleanup all managed SSBOs.
     */
    public static synchronized void cleanupAll() {
        if (instance == null) return;

        LOGGER.info("WorldGenEventHandler.cleanupAll() — cleaning up {} levels", activeLevels.size());
        for (ShaderSSBOManager manager : activeLevels.values()) {
            try {
                manager.cleanup();
            } catch (Exception e) {
                LOGGER.error("Error during SSBO cleanup", e);
            }
        }
        activeLevels.clear();
    }

    // ============================================================================
    // Reflection Utilities
    // ============================================================================

    private static Class<?> loadClassOrNull(String className) {
        try {
            return Class.forName(className);
        } catch (Throwable t) {
            // Catch Errors and Exceptions (e.g., ExceptionInInitializerError) to avoid
            // failing unit tests or running outside of a full Minecraft bootstrap.
            return null;
        }
    }

    private static Method findMethodOrNull(Class<?> clazz, String methodName) {
        if (clazz == null) return null;
        try {
            for (Method m : clazz.getDeclaredMethods()) {
                if (m.getName().equals(methodName)) {
                    m.setAccessible(true);
                    return m;
                }
            }
        } catch (Exception ignored) {}
        return null;
    }
}

