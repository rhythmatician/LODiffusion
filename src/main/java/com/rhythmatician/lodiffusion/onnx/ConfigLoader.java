package com.rhythmatician.lodiffusion.onnx;

import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

/**
 * Loads ModelConfig from a JSON file produced alongside ONNX models.
 */
public final class ConfigLoader {

    private static final Logger LOGGER = Logger.getLogger(ConfigLoader.class.getName());
    private static final Gson GSON = new GsonBuilder().create();

    private ConfigLoader() {}

    public static ModelConfig load(Path jsonPath) throws IOException {
        try (var reader = new InputStreamReader(Files.newInputStream(jsonPath))) {
            ModelConfig config = GSON.fromJson(reader, ModelConfig.class);
            if (config == null) {
                throw new IOException("Failed to parse config: " + jsonPath);
            }
            config.validate();
            LOGGER.info("Loaded model config: " + config.modelName() + " from " + jsonPath);
            return config;
        }
    }
}
