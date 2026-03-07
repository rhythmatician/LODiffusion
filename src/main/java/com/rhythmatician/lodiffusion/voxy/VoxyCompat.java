package com.rhythmatician.lodiffusion.voxy;

import java.lang.reflect.Method;
import java.util.logging.Logger;

/**
 * Runtime compatibility layer for Voxy.
 *
 * <p>Voxy has no public modding API, so we detect its presence via class loading
 * and access its internals through reflection.  All Voxy interactions go through
 * this class so that breakage from Voxy updates is contained.
 *
 * <p><b>Key Voxy concepts:</b>
 * <ul>
 *   <li>{@code WorldEngine} — the root object holding block/biome mappings and
 *       the section storage backend (RocksDB).</li>
 *   <li>{@code VoxelizedSection} — a 16³ ingest chunk with a built-in mip pyramid
 *       (16³ + 8³ + 4³ + 2³ + 1 = 4681 packed voxels).</li>
 *   <li>{@code WorldUpdater.insertUpdate(engine, section)} — the blocking call
 *       that pushes a section into the world, propagating through LOD layers.</li>
 *   <li>{@code Mapper} — translates Minecraft {@code BlockState}/{@code Biome}
 *       into Voxy's compact 64-bit voxel encoding.</li>
 * </ul>
 */
public final class VoxyCompat {

    private static final Logger LOGGER = Logger.getLogger(VoxyCompat.class.getName());

    /** Cached availability flag — computed once at first access. */
    private static volatile Boolean available;

    // Reflected classes (resolved lazily)
    private static Class<?> worldEngineClass;
    private static Class<?> worldUpdaterClass;
    private static Class<?> voxelizedSectionClass;
    private static Class<?> mapperClass;

    // Reflected methods
    private static Method insertUpdateMethod;      // WorldUpdater.insertUpdate(WorldEngine, VoxelizedSection)
    private static Method createEmptyMethod;        // VoxelizedSection.createEmpty()
    private static Method getMapperMethod;          // WorldEngine.getMapper()
    private static Method mipSectionMethod;         // WorldConversionFactory.mipSection(VoxelizedSection, Mapper)
    private static Method ofEngineMethod;           // WorldIdentifier.ofEngine(World)
    private static Method ofEngineNullableMethod;   // WorldIdentifier.ofEngineNullable(World)

    private VoxyCompat() {}

    // ------------------------------------------------------------------ //
    //  Detection
    // ------------------------------------------------------------------ //

    /** True if Voxy classes are on the classpath. */
    public static boolean isAvailable() {
        Boolean cached = available;
        if (cached != null) return cached;

        synchronized (VoxyCompat.class) {
            if (available != null) return available;
            try {
                worldEngineClass      = Class.forName("me.cortex.voxy.common.world.WorldEngine");
                worldUpdaterClass     = Class.forName("me.cortex.voxy.common.world.WorldUpdater");
                voxelizedSectionClass = Class.forName("me.cortex.voxy.common.voxelization.VoxelizedSection");
                mapperClass           = Class.forName("me.cortex.voxy.common.world.other.Mapper");

                // Resolve key methods
                insertUpdateMethod = worldUpdaterClass.getMethod("insertUpdate",
                        worldEngineClass, voxelizedSectionClass);
                createEmptyMethod  = voxelizedSectionClass.getMethod("createEmpty");
                getMapperMethod    = worldEngineClass.getMethod("getMapper");

                Class<?> convFactoryClass = Class.forName(
                        "me.cortex.voxy.common.voxelization.WorldConversionFactory");
                mipSectionMethod = convFactoryClass.getMethod("mipSection",
                        voxelizedSectionClass, mapperClass);

                // WorldIdentifier — for obtaining WorldEngine from a World
                Class<?> worldIdClass = Class.forName("me.cortex.voxy.commonImpl.WorldIdentifier");
                ofEngineMethod = worldIdClass.getMethod("ofEngine",
                        net.minecraft.world.World.class);
                ofEngineNullableMethod = worldIdClass.getMethod("ofEngineNullable",
                        net.minecraft.world.World.class);

                available = true;
                LOGGER.info("Voxy detected — reflection bindings resolved");
            } catch (ClassNotFoundException | NoSuchMethodException e) {
                available = false;
                LOGGER.info("Voxy not found: " + e.getMessage());
            }
            return available;
        }
    }

    // ------------------------------------------------------------------ //
    //  Accessors for reflected types
    // ------------------------------------------------------------------ //

    /** Create an empty {@code VoxelizedSection} (16³ + mip pyramid). */
    public static Object createEmptySection() {
        ensureAvailable();
        try {
            return createEmptyMethod.invoke(null);
        } catch (Exception e) {
            throw new RuntimeException("Failed to create VoxelizedSection", e);
        }
    }

    /** Get the Mapper from a WorldEngine instance. */
    public static Object getMapper(Object worldEngine) {
        ensureAvailable();
        try {
            return getMapperMethod.invoke(worldEngine);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get Mapper from WorldEngine", e);
        }
    }

    /** Compute the mip pyramid for a VoxelizedSection. */
    public static void mipSection(Object section, Object mapper) {
        ensureAvailable();
        try {
            mipSectionMethod.invoke(null, section, mapper);
        } catch (Exception e) {
            throw new RuntimeException("Failed to mip section", e);
        }
    }

    /** Insert a VoxelizedSection into a WorldEngine (blocking). */
    public static void insertUpdate(Object worldEngine, Object section) {
        ensureAvailable();
        try {
            insertUpdateMethod.invoke(null, worldEngine, section);
        } catch (Exception e) {
            throw new RuntimeException("Failed to insert section into Voxy world", e);
        }
    }

    /**
     * Get the Voxy WorldEngine for a given Minecraft World.
     *
     * <p>Uses {@code WorldIdentifier.ofEngineNullable(World)} which returns null
     * if Voxy hasn't created the engine for this world yet.
     *
     * @param world the Minecraft World instance
     * @return the WorldEngine, or null if not yet available
     */
    public static Object getWorldEngine(net.minecraft.world.World world) {
        ensureAvailable();
        try {
            return ofEngineNullableMethod.invoke(null, world);
        } catch (Exception e) {
            LOGGER.warning("Failed to get WorldEngine: " + e.getMessage());
            return null;
        }
    }

    /**
     * Get the Voxy WorldEngine for a given Minecraft World, creating it if needed.
     *
     * @param world the Minecraft World instance
     * @return the WorldEngine (never null if Voxy is available)
     */
    public static Object getOrCreateWorldEngine(net.minecraft.world.World world) {
        ensureAvailable();
        try {
            return ofEngineMethod.invoke(null, world);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get/create WorldEngine", e);
        }
    }

    // ------------------------------------------------------------------ //
    //  VoxelizedSection field access
    // ------------------------------------------------------------------ //

    /** Set the position fields (x, y, z) on a VoxelizedSection. */
    public static void setSectionPosition(Object section, int x, int y, int z) {
        try {
            var xField = voxelizedSectionClass.getField("x");
            var yField = voxelizedSectionClass.getField("y");
            var zField = voxelizedSectionClass.getField("z");
            xField.setInt(section, x);
            yField.setInt(section, y);
            zField.setInt(section, z);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set section position", e);
        }
    }

    /** Get the raw voxel data array from a VoxelizedSection. */
    public static long[] getSectionData(Object section) {
        try {
            var field = voxelizedSectionClass.getField("section");
            return (long[]) field.get(section);
        } catch (Exception e) {
            throw new RuntimeException("Failed to get section data", e);
        }
    }

    /** Set the L0 non-air count on a VoxelizedSection. */
    public static void setNonAirCount(Object section, int count) {
        try {
            var field = voxelizedSectionClass.getField("lvl0NonAirCount");
            field.setInt(section, count);
        } catch (Exception e) {
            throw new RuntimeException("Failed to set non-air count", e);
        }
    }

    // ------------------------------------------------------------------ //
    //  Voxy 64-bit voxel encoding helpers
    // ------------------------------------------------------------------ //

    /** Voxel bit layout constants matching Voxy's Mapper. */
    public static final int BLOCK_ID_SHIFT = 27;
    public static final int BLOCK_ID_BITS  = 20;
    public static final long BLOCK_ID_MASK = ((1L << BLOCK_ID_BITS) - 1) << BLOCK_ID_SHIFT;

    public static final int BIOME_ID_SHIFT = 47;
    public static final int BIOME_ID_BITS  = 9;
    public static final long BIOME_ID_MASK = ((1L << BIOME_ID_BITS) - 1) << BIOME_ID_SHIFT;

    public static final int LIGHT_SHIFT = 56;

    /** Compose a 64-bit voxel value from block ID, biome ID, and light. */
    public static long composeVoxel(int blockId, int biomeId, int light) {
        return ((long) light << LIGHT_SHIFT)
             | ((long)(biomeId & 0x1FF) << BIOME_ID_SHIFT)
             | ((long)(blockId & ((1 << BLOCK_ID_BITS) - 1)) << BLOCK_ID_SHIFT);
    }

    /** True if the voxel is air (block ID field is zero). */
    public static boolean isAir(long voxel) {
        return (voxel & BLOCK_ID_MASK) == 0;
    }

    /** L0 index into VoxelizedSection.section[] for (x, y, z) in [0,15]. */
    public static int l0Index(int x, int y, int z) {
        return (y << 8) | (z << 4) | x;
    }

    // ------------------------------------------------------------------ //
    //  Internal
    // ------------------------------------------------------------------ //

    private static void ensureAvailable() {
        if (!isAvailable()) {
            throw new IllegalStateException("Voxy is not available");
        }
    }
}
