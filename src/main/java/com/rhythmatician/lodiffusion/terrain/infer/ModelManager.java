package com.rhythmatician.lodiffusion.terrain.infer;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

import com.rhythmatician.lodiffusion.Config;

import ai.djl.MalformedModelException;
import ai.djl.Model;

/**
 * Thread-safe singleton for managing the global ONNX model instance.
 * Loads model lazily on first access and provides cleanup on shutdown.
 */
public final class ModelManager {
    private static final Object LOCK = new Object();
    private static volatile Model model;
    private static volatile boolean shutdownHookRegistered = false;

    private ModelManager() {}

    /**
     * Get the loaded model, loading it if necessary.
     * @return The DJL Model instance
     * @throws IllegalStateException if model cannot be loaded
     */
    public static Model getOrLoad() {
        Model m = model;
        if (m != null) return m;
        synchronized (LOCK) {
            if (model == null) {
                model = load(Config.modelPath());
                registerShutdownHook();
            }
            return model;
        }
    }

    private static Model load(Path path) {
        if (!Files.isRegularFile(path)) {
            throw new IllegalStateException("ONNX model file not found: " + path);
        }
        
        try {
            Model m = Model.newInstance("lodiffusion-onnx", "OnnxRuntime");
            m.load(path);
            return m;
        } catch (IOException e) {
            throw new IllegalStateException("Failed to load ONNX model: " + path, e);
        } catch (MalformedModelException e) {
            throw new IllegalStateException("Invalid ONNX model format: " + path, e);
        }
    }

    private static void registerShutdownHook() {
        if (!shutdownHookRegistered) {
            Runtime.getRuntime().addShutdownHook(new Thread(ModelManager::close, "ModelManager-Shutdown"));
            shutdownHookRegistered = true;
        }
    }

    /**
     * Close the model and clean up resources.
     * Safe to call multiple times.
     */
    public static void close() {
        Model m = model;
        if (m != null) {
            synchronized (LOCK) {
                if (model != null) {
                    model.close();
                    model = null;
                }
            }
        }
    }

    /**
     * Check if model is available without triggering load.
     * @return true if model file exists at configured path
     */
    public static boolean isAvailable() {
        return Files.isRegularFile(Config.modelPath());
    }
}
