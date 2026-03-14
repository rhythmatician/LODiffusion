package io.github.lodiffusion.worldgen;

import org.apache.logging.log4j.LogManager;
import org.apache.logging.log4j.Logger;

import java.lang.reflect.Array;
import java.lang.reflect.Field;
import java.lang.reflect.InvocationHandler;
import java.lang.reflect.Method;
import java.lang.reflect.Proxy;
import java.nio.FloatBuffer;
import java.nio.IntBuffer;
import java.util.IdentityHashMap;
import java.util.List;
import java.util.Map;
import java.util.ArrayList;

/**
 * Reflection-driven extractor that walks a Minecraft NoiseRouter and serializes
 * its noise parameters into GPU SSBO buffers.
 *
 * This class is intentionally written without any direct references to Minecraft
 * classes (net.minecraft.*) so that it compiles even if the Minecraft dependency
 * is not present at compile time.
 */
public class NoiseRouterExtractor {
    private static final Logger LOGGER = LogManager.getLogger();

    // ============================================================================
    // SSBO Layout Constants (must match GLSL shader definitions)
    // ============================================================================

    private static final int IMPROVED_ORIGINS_BINDING = 0;
    private static final int IMPROVED_PERMS_BINDING = 1;
    private static final int IMPROVED_PERMS_STRIDE = 256;

    private static final int PERLIN_INT_BINDING = 2;
    private static final int PERLIN_FLOAT_BINDING = 3;
    private static final int MAX_OCTAVES = 16;

    private static final int NORMAL_NOISE_INT_BINDING = 4;
    private static final int NORMAL_NOISE_FLOAT_BINDING = 5;

    private static final int SPLINE_DATA_BINDING = 6;
    private static final int DENSITY_OUTPUT_BINDING = 7;

    // ============================================================================
    // Reflection type caches (loaded lazily)
    // ============================================================================

    private final Class<?> densityFunctionClass;
    private final Class<?> densityFunctionVisitorClass;
    private final Class<?> densityFunctionsNoiseClass;
    private final Class<?> densityFunctionsSplineClass;
    private final Class<?> densityFunctionsMarkerClass;
    private final Class<?> densityFunctionNoiseHolderClass;

    private final Class<?> normalNoiseClass;
    private final Class<?> perlinNoiseClass;
    private final Class<?> improvedNoiseClass;
    private final Class<?> cubicSplineClass;

    // Cached reflection methods (for performance)
    private final Method mapAllMethod;
    private final Method noiseHolderNoiseMethod;
    private final Method cubicSplineControlPointsMethod;

    // ============================================================================
    // Extraction state
    // ============================================================================

    /** NormalNoise instance → GPU index */
    private final Map<Object, Integer> noiseIndexMap = new IdentityHashMap<>();

    /** ImprovedNoise instance → GPU index */
    private final Map<Object, Integer> improvedNoiseIndexMap = new IdentityHashMap<>();

    /** PerlinNoise instance → GPU index */
    private final Map<Object, Integer> perlinNoiseIndexMap = new IdentityHashMap<>();

    /** Discovered ImprovedNoise instances (retains insertion order) */
    private final List<Object> improvedNoises = new ArrayList<>();

    /** Discovered PerlinNoise instances */
    private final List<Object> perlinNoises = new ArrayList<>();

    /** Discovered NormalNoise instances */
    private final List<Object> normalNoises = new ArrayList<>();

    /** Spline control point buffer and offsets */
    private final List<Float> splineDataFloats = new ArrayList<>();
    private final Map<Object, Integer> splineOffsets = new IdentityHashMap<>();

    // ----------------------------------------------------------------------------------------------------------------
    // Constructor (builds reflection metadata)
    // ----------------------------------------------------------------------------------------------------------------

    public NoiseRouterExtractor() {
        this.densityFunctionClass = loadClassOrNull("net.minecraft.world.level.levelgen.DensityFunction");
        this.densityFunctionVisitorClass = loadClassOrNull("net.minecraft.world.level.levelgen.DensityFunction$Visitor");
        this.densityFunctionsNoiseClass = loadClassOrNull("net.minecraft.world.level.levelgen.DensityFunctions$Noise");
        this.densityFunctionsSplineClass = loadClassOrNull("net.minecraft.world.level.levelgen.DensityFunctions$Spline");
        this.densityFunctionsMarkerClass = loadClassOrNull("net.minecraft.world.level.levelgen.DensityFunctions$Marker");
        this.densityFunctionNoiseHolderClass = loadClassOrNull("net.minecraft.world.level.levelgen.DensityFunction$NoiseHolder");

        this.normalNoiseClass = loadClassOrNull("net.minecraft.world.level.levelgen.synth.NormalNoise");
        this.perlinNoiseClass = loadClassOrNull("net.minecraft.world.level.levelgen.synth.PerlinNoise");
        this.improvedNoiseClass = loadClassOrNull("net.minecraft.world.level.levelgen.synth.ImprovedNoise");
        this.cubicSplineClass = loadClassOrNull("net.minecraft.util.CubicSpline");

        this.mapAllMethod = findMethodOrNull("mapAll", Object.class);
        this.noiseHolderNoiseMethod = findMethodOrNull(densityFunctionNoiseHolderClass, "noise");
        this.cubicSplineControlPointsMethod = findMethodOrNull(cubicSplineClass, "getControlPoints");
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Public API
    // ----------------------------------------------------------------------------------------------------------------

    /**
     * Extracts SSBO-compatible buffers from a runtime NoiseRouter instance.
     *
     * @param noiseRouter A runtime instance of net.minecraft.world.level.levelgen.NoiseRouter
     */
    public NoiseRouterData extract(Object noiseRouter) {
        LOGGER.info("Starting NoiseRouter extraction...");
        if (noiseRouter == null) {
            throw new IllegalArgumentException("noiseRouter must not be null");
        }
        if (mapAllMethod == null || densityFunctionVisitorClass == null) {
            throw new IllegalStateException("Unable to locate DensityFunction.mapAll or Visitor interface at runtime");
        }

        Object visitor = createVisitorProxy();
        try {
            mapAllMethod.invoke(noiseRouter, visitor);
        } catch (Exception e) {
            throw new RuntimeException("Failed to invoke mapAll() on NoiseRouter", e);
        }

        LOGGER.info("Discovered {} ImprovedNoise instances", improvedNoises.size());
        LOGGER.info("Discovered {} PerlinNoise instances", perlinNoises.size());
        LOGGER.info("Discovered {} NormalNoise instances", normalNoises.size());

        NoiseRouterData data = new NoiseRouterData();
        data.improvedOrigins = extractImprovedOrigins();
        data.improvedPerms = extractImprovedPerms();
        data.perlinInts = extractPerlinInts();
        data.perlinFloats = extractPerlinFloats();
        data.normalNoiseInts = extractNormalNoiseInts();
        data.normalNoiseFloats = extractNormalNoiseFloats();

        float[] splineArray = new float[splineDataFloats.size()];
        for (int i = 0; i < splineDataFloats.size(); i++) {
            splineArray[i] = splineDataFloats.get(i);
        }
        data.splineData = FloatBuffer.wrap(splineArray);

        // Second pass: resolve named noise indices from specific NoiseRouter fields
        wireNamedIndices(noiseRouter, data);

        LOGGER.info("NoiseRouter extraction complete. Spline data size: {} floats", data.splineData.capacity());
        LOGGER.info("Named indices: continents={} erosion={} ridges={} shift={}",
                data.nnContinents, data.nnErosion, data.nnRidges, data.shiftNoiseIndex);
        return data;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Visitor Proxy
    // ----------------------------------------------------------------------------------------------------------------

    private Object createVisitorProxy() {
        InvocationHandler handler = (proxy, method, args) -> {
            if (args != null && args.length == 1) {
                Object function = args[0];
                unwrapAndProcess(function);
                return function; // match DensityFunction.Visitor.apply()
            }
            // Fallback: return null for any other method
            return null;
        };

        return Proxy.newProxyInstance(
            getClass().getClassLoader(),
            new Class<?>[]{densityFunctionVisitorClass},
            handler
        );
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Graph traversal and node processing
    // ----------------------------------------------------------------------------------------------------------------

    private void unwrapAndProcess(Object function) {
        if (function == null) return;

        if (densityFunctionsNoiseClass != null && densityFunctionsNoiseClass.isInstance(function)) {
            processNoise(function);
            return;
        }

        if (densityFunctionsSplineClass != null && densityFunctionsSplineClass.isInstance(function)) {
            processSpline(function);
            return;
        }

        if (densityFunctionsMarkerClass != null && densityFunctionsMarkerClass.isInstance(function)) {
            Object underlying = getMarkerArgument(function);
            if (underlying != null) {
                unwrapAndProcess(underlying);
            }
            return;
        }

        // Handle NoiseHolder directly — visited via visitNoise() from ShiftedNoise/ShiftA/ShiftB.
        // This registers the NormalNoise for continents, erosion, ridges, and the SHIFT noise.
        if (densityFunctionNoiseHolderClass != null && densityFunctionNoiseHolderClass.isInstance(function)) {
            processNoiseHolder(function);
            return;
        }

        // Fallback: if this object has a field named "wrapped" or "argument", try to unwrap
        Object fallback = tryUnwrapByName(function, "wrapped");
        if (fallback == null) fallback = tryUnwrapByName(function, "argument");
        if (fallback != null) {
            unwrapAndProcess(fallback);
        }
    }

    private void processNoiseHolder(Object holder) {
        try {
            if (noiseHolderNoiseMethod == null) return;
            Object noiseObj = noiseHolderNoiseMethod.invoke(holder);
            if (noiseObj == null || normalNoiseClass == null || !normalNoiseClass.isInstance(noiseObj)) return;

            if (!noiseIndexMap.containsKey(noiseObj)) {
                int index = normalNoises.size();
                noiseIndexMap.put(noiseObj, index);
                normalNoises.add(noiseObj);

                Object first  = getFieldValue(noiseObj, "first");
                Object second = getFieldValue(noiseObj, "second");
                registerPerlinNoise(first);
                registerPerlinNoise(second);
            }
        } catch (Exception e) {
            LOGGER.warn("Failed to process NoiseHolder node", e);
        }
    }

    private void processNoise(Object noiseFunc) {
        try {
            Object holder = getNoiseHolder(noiseFunc);
            if (holder == null) return;

            Object noiseObj = noiseHolderNoiseMethod != null ? noiseHolderNoiseMethod.invoke(holder) : null;
            if (noiseObj == null || normalNoiseClass == null || !normalNoiseClass.isInstance(noiseObj)) {
                return;
            }

            if (!noiseIndexMap.containsKey(noiseObj)) {
                int index = normalNoises.size();
                noiseIndexMap.put(noiseObj, index);
                normalNoises.add(noiseObj);

                Object first = getFieldValue(noiseObj, "first");
                Object second = getFieldValue(noiseObj, "second");

                registerPerlinNoise(first);
                registerPerlinNoise(second);
            }
        } catch (Exception e) {
            LOGGER.warn("Failed to process Noise node", e);
        }
    }

    private void processSpline(Object splineFunc) {
        if (cubicSplineClass == null) return;

        try {
            Object spline = getFieldValue(splineFunc, "spline");
            if (spline != null && !splineOffsets.containsKey(spline)) {
                int offset = splineDataFloats.size();
                splineOffsets.put(spline, offset);

                // TODO: Flatten spline control points.
                // For now, we reserve the offset so shaders can bind a predictable index.
            }
        } catch (Exception e) {
            LOGGER.warn("Failed to process Spline node", e);
        }
    }

    private void registerPerlinNoise(Object pn) {
        if (pn == null || perlinNoiseClass == null || !perlinNoiseClass.isInstance(pn)) return;
        if (perlinNoiseIndexMap.containsKey(pn)) return;

        int index = perlinNoises.size();
        perlinNoiseIndexMap.put(pn, index);
        perlinNoises.add(pn);

        Object[] octaves = getImprovedOctaves(pn);
        if (octaves == null) return;

        for (Object improved : octaves) {
            if (improved != null && !improvedNoiseIndexMap.containsKey(improved)) {
                int improvedIdx = improvedNoises.size();
                improvedNoiseIndexMap.put(improved, improvedIdx);
                improvedNoises.add(improved);
            }
        }
    }

    // ----------------------------------------------------------------------------------------------------------------
    // SSBO extraction helpers
    // ----------------------------------------------------------------------------------------------------------------

    private FloatBuffer extractImprovedOrigins() {
        FloatBuffer buffer = FloatBuffer.allocate(improvedNoises.size() * 3);
        for (Object improved : improvedNoises) {
            try {
                double xo = getDoubleField(improved, "xo");
                double yo = getDoubleField(improved, "yo");
                double zo = getDoubleField(improved, "zo");
                buffer.put((float) xo);
                buffer.put((float) yo);
                buffer.put((float) zo);
            } catch (Exception e) {
                LOGGER.warn("Failed to extract origin from ImprovedNoise", e);
                buffer.put(0).put(0).put(0);
            }
        }
        buffer.flip();
        return buffer;
    }

    private IntBuffer extractImprovedPerms() {
        IntBuffer buffer = IntBuffer.allocate(improvedNoises.size() * IMPROVED_PERMS_STRIDE);
        for (Object improved : improvedNoises) {
            try {
                byte[] perms = (byte[]) getFieldValue(improved, "p");
                for (byte perm : perms) {
                    buffer.put(perm & 0xFF);
                }
            } catch (Exception e) {
                LOGGER.warn("Failed to extract permutations from ImprovedNoise", e);
                for (int i = 0; i < IMPROVED_PERMS_STRIDE; i++) {
                    buffer.put(i);
                }
            }
        }
        buffer.flip();
        return buffer;
    }

    private IntBuffer extractPerlinInts() {
        IntBuffer buffer = IntBuffer.allocate(perlinNoises.size() * (1 + MAX_OCTAVES));
        for (Object pn : perlinNoises) {
            try {
                int firstOctave = getIntField(pn, "firstOctave");
                buffer.put(firstOctave);

                Object[] octaves = getImprovedOctaves(pn);
                for (int i = 0; i < MAX_OCTAVES; i++) {
                    if (octaves != null && i < octaves.length && octaves[i] != null) {
                        Integer idx = improvedNoiseIndexMap.get(octaves[i]);
                        buffer.put(idx != null ? idx : -1);
                    } else {
                        buffer.put(-1);
                    }
                }
            } catch (Exception e) {
                LOGGER.warn("Failed to extract PerlinNoise int data", e);
                buffer.put(0);
                for (int i = 0; i < MAX_OCTAVES; i++) {
                    buffer.put(-1);
                }
            }
        }
        buffer.flip();
        return buffer;
    }

    private FloatBuffer extractPerlinFloats() {
        FloatBuffer buffer = FloatBuffer.allocate(perlinNoises.size() * (2 + MAX_OCTAVES));
        for (Object pn : perlinNoises) {
            try {
                double lowestFreqFactor = getDoubleField(pn, "lowestFreqInputFactor");
                double lowestValFactor = getDoubleField(pn, "lowestFreqValueFactor");
                buffer.put((float) lowestFreqFactor);
                buffer.put((float) lowestValFactor);

                Object amplitudes = getFieldValue(pn, "amplitudes");
                int amplitudeCount = 0;
                double[] amplitudeArray = new double[MAX_OCTAVES];

                if (amplitudes != null) {
                    try {
                        Method sizeMethod = amplitudes.getClass().getMethod("size");
                        int size = (int) sizeMethod.invoke(amplitudes);
                        amplitudeCount = Math.min(size, MAX_OCTAVES);
                        Method getDoubleMethod = amplitudes.getClass().getMethod("getDouble", int.class);
                        for (int i = 0; i < amplitudeCount; i++) {
                            amplitudeArray[i] = (double) getDoubleMethod.invoke(amplitudes, i);
                        }
                    } catch (Exception ex) {
                        LOGGER.warn("Failed to extract amplitudes from DoubleList", ex);
                    }
                }
                for (int i = 0; i < MAX_OCTAVES; i++) {
                    buffer.put(i < amplitudeCount ? (float) amplitudeArray[i] : 0.0f);
                }
            } catch (Exception e) {
                LOGGER.warn("Failed to extract PerlinNoise float data", e);
                buffer.put(1.0f).put(1.0f);
                for (int i = 0; i < MAX_OCTAVES; i++) {
                    buffer.put(0.0f);
                }
            }
        }
        buffer.flip();
        return buffer;
    }

    private IntBuffer extractNormalNoiseInts() {
        IntBuffer buffer = IntBuffer.allocate(normalNoises.size() * 2);
        for (Object nn : normalNoises) {
            try {
                Object first = getFieldValue(nn, "first");
                Object second = getFieldValue(nn, "second");
                Integer firstIdx = perlinNoiseIndexMap.get(first);
                Integer secondIdx = perlinNoiseIndexMap.get(second);
                buffer.put(firstIdx != null ? firstIdx : -1);
                buffer.put(secondIdx != null ? secondIdx : -1);
            } catch (Exception e) {
                LOGGER.warn("Failed to extract NormalNoise int data", e);
                buffer.put(-1).put(-1);
            }
        }
        buffer.flip();
        return buffer;
    }

    private FloatBuffer extractNormalNoiseFloats() {
        FloatBuffer buffer = FloatBuffer.allocate(normalNoises.size());
        for (Object nn : normalNoises) {
            try {
                double valueFactor = getDoubleField(nn, "valueFactor");
                buffer.put((float) valueFactor);
            } catch (Exception e) {
                LOGGER.warn("Failed to extract NormalNoise float data", e);
                buffer.put(1.0f);
            }
        }
        buffer.flip();
        return buffer;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Reflection helpers
    // ----------------------------------------------------------------------------------------------------------------

    private Class<?> loadClassOrNull(String className) {
        try {
            return Class.forName(className);
        } catch (ClassNotFoundException e) {
            return null;
        }
    }

    private Method findMethodOrNull(String methodName, Class<?>... paramTypes) {
        if (densityFunctionClass == null) return null;
        try {
            return densityFunctionClass.getMethod(methodName, paramTypes);
        } catch (Exception ignored) {
            return null;
        }
    }

    private Method findMethodOrNull(Class<?> clazz, String methodName, Class<?>... paramTypes) {
        if (clazz == null) return null;
        try {
            return clazz.getMethod(methodName, paramTypes);
        } catch (Exception ignored) {
            return null;
        }
    }

    private Object getFieldValue(Object obj, String fieldName) {
        if (obj == null) return null;
        try {
            Field field = obj.getClass().getDeclaredField(fieldName);
            field.setAccessible(true);
            return field.get(obj);
        } catch (Exception ignored) {
            // Ignore missing fields; we'll attempt other strategies
            return null;
        }
    }

    private double getDoubleField(Object obj, String fieldName) throws Exception {
        Object value = getFieldValue(obj, fieldName);
        if (value instanceof Number) {
            return ((Number) value).doubleValue();
        }
        throw new IllegalStateException("Field " + fieldName + " is not numeric");
    }

    private int getIntField(Object obj, String fieldName) throws Exception {
        Object value = getFieldValue(obj, fieldName);
        if (value instanceof Number) {
            return ((Number) value).intValue();
        }
        throw new IllegalStateException("Field " + fieldName + " is not numeric");
    }

    private Object getNoiseHolder(Object noiseFunc) {
        return getFieldValue(noiseFunc, "noiseData");
    }

    private Object getMarkerArgument(Object marker) {
        return getFieldValue(marker, "wrapped");
    }

    private Object[] getImprovedOctaves(Object perlinNoise) {
        Object value = getFieldValue(perlinNoise, "noiseLevels");
        if (value == null) return null;
        if (value.getClass().isArray()) {
            int len = Array.getLength(value);
            Object[] result = new Object[len];
            for (int i = 0; i < len; i++) {
                result[i] = Array.get(value, i);
            }
            return result;
        }
        return null;
    }

    private Object tryUnwrapByName(Object obj, String fieldName) {
        Object candidate = getFieldValue(obj, fieldName);
        if (candidate != null) return candidate;
        return null;
    }

    // ----------------------------------------------------------------------------------------------------------------
    // Named index wiring (second pass after mapAll() traversal)
    // ----------------------------------------------------------------------------------------------------------------

    /**
     * Populates named noise indices in {@code data} by walking specific fields of the NoiseRouter.
     * Requires that {@code mapAll()} has already been run so {@code noiseIndexMap} is populated.
     */
    private void wireNamedIndices(Object noiseRouter, NoiseRouterData data) {
        data.nnContinents    = indexForShiftedNoise(noiseRouter, "continents");
        data.nnErosion       = indexForShiftedNoise(noiseRouter, "erosion");
        data.nnRidges        = indexForShiftedNoise(noiseRouter, "ridges");
        data.shiftNoiseIndex = indexForShiftNoise(noiseRouter, "continents");
        // nnDepthNoise and nnJagged are buried deep in finalDensity — tracked separately (WS-1.2 ext.)

        LOGGER.info("Named noise indices: continents={}, erosion={}, ridges={}, shift={}",
                data.nnContinents, data.nnErosion, data.nnRidges, data.shiftNoiseIndex);
    }

    /**
     * Returns the NormalNoise SSBO index for the noise wrapped inside a ShiftedNoise2d field.
     * noiseRouter.{fieldName}() → ShiftedNoise → .noise (NoiseHolder) → NormalNoise → index
     */
    private int indexForShiftedNoise(Object noiseRouter, String fieldName) {
        try {
            Object shiftedNoise = invokeAccessor(noiseRouter, fieldName);
            if (shiftedNoise == null) return -1;

            // ShiftedNoise.noise is a DensityFunction$NoiseHolder
            Object noiseHolder = getFieldValue(shiftedNoise, "noise");
            if (noiseHolder == null) return -1;

            Object normalNoise = noiseHolderNoiseMethod != null ? noiseHolderNoiseMethod.invoke(noiseHolder) : null;
            if (normalNoise == null) return -1;

            Integer idx = noiseIndexMap.get(normalNoise);
            return idx != null ? idx : -1;
        } catch (Exception e) {
            LOGGER.warn("Failed to determine index for ShiftedNoise '{}'", fieldName, e);
            return -1;
        }
    }

    /**
     * Returns the SHIFT NormalNoise SSBO index by walking:
     *   noiseRouter.{fieldName}() (ShiftedNoise) → .shiftX (ShiftA) → .offsetNoise (NoiseHolder) → NormalNoise
     *
     * Both ShiftA and ShiftB use the same underlying Noises.SHIFT NormalNoise.
     * The coord permutation (bx,0,bz vs bz,bx,0) is applied at call time in the shader,
     * so a single SSBO index covers both shift_x and shift_z.
     */
    private int indexForShiftNoise(Object noiseRouter, String fieldName) {
        try {
            Object shiftedNoise = invokeAccessor(noiseRouter, fieldName);
            if (shiftedNoise == null) return -1;

            // ShiftedNoise.shiftX is a ShiftA instance
            Object shiftA = getFieldValue(shiftedNoise, "shiftX");
            if (shiftA == null) return -1;

            // ShiftA.offsetNoise is a DensityFunction$NoiseHolder
            Object offsetNoiseHolder = getFieldValue(shiftA, "offsetNoise");
            if (offsetNoiseHolder == null) return -1;

            Object normalNoise = noiseHolderNoiseMethod != null ? noiseHolderNoiseMethod.invoke(offsetNoiseHolder) : null;
            if (normalNoise == null) return -1;

            Integer idx = noiseIndexMap.get(normalNoise);
            return idx != null ? idx : -1;
        } catch (Exception e) {
            LOGGER.warn("Failed to determine shift noise index from '{}'", fieldName, e);
            return -1;
        }
    }

    /** Invokes a no-arg accessor method by name on the given object. */
    private Object invokeAccessor(Object obj, String methodName) {
        try {
            java.lang.reflect.Method m = obj.getClass().getMethod(methodName);
            return m.invoke(obj);
        } catch (Exception e) {
            return null;
        }
    }

    // ============================================================================
    // Output Data Container
    // ============================================================================

    public static class NoiseRouterData {
        public FloatBuffer improvedOrigins;
        public IntBuffer improvedPerms;
        public IntBuffer perlinInts;
        public FloatBuffer perlinFloats;
        public IntBuffer normalNoiseInts;
        public FloatBuffer normalNoiseFloats;
        public FloatBuffer splineData;

        // Named NormalNoise indices within the flat SSBO arrays (binding 4/5).
        // -1 means not found — shader will use its fallback path.
        public int nnContinents    = -1;
        public int nnErosion       = -1;
        public int nnRidges        = -1;
        public int nnDepthNoise    = -1;  // TODO: extract from finalDensity tree (WS-1.2 ext.)
        public int nnJagged        = -1;  // TODO: extract from finalDensity tree (WS-1.2 ext.)
        public int shiftNoiseIndex = -1;  // Noises.SHIFT — same index for both ShiftA and ShiftB

        public void uploadToGPU() {
            LOGGER.info("SSBO upload requested (not yet implemented)");
        }
    }
}
