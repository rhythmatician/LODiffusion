package com.rhythmatician.lodiffusion.onnx;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.logging.Logger;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import com.google.gson.JsonArray;
import com.google.gson.JsonElement;
import com.google.gson.JsonObject;
import com.google.gson.JsonParser;

/**
 * Loads ModelConfig from a JSON file produced alongside ONNX models.
 */
public final class ConfigLoader {

    private static final Logger LOGGER = Logger.getLogger(ConfigLoader.class.getName());
    private static final Gson GSON = new GsonBuilder().create();

    private ConfigLoader() {}

    public static ModelConfig load(Path jsonPath) throws IOException {
        // Read the JSON as a tree first to allow pre-processing (e.g., symbolic dims)
        String json = Files.readString(jsonPath);
        JsonObject root = JsonParser.parseString(json).getAsJsonObject();

    // Normalize common snake_case top-level keys to Java record field names
    renameKey(root, "model_name", "modelName");
    renameKey(root, "optional_inputs", "optionalInputs");
    renameKey(root, "block_palette", "blockPalette");

    // If outputs.block_logits has a symbolic dimension (e.g., "N_blocks"),
        // replace it with the concrete value from block_palette.size before deserialization.
        try {
            JsonObject paletteObj = null;
            if (root.has("blockPalette")) {
                paletteObj = root.getAsJsonObject("blockPalette");
            } else if (root.has("block_palette")) {
                paletteObj = root.getAsJsonObject("block_palette");
            }

            if (paletteObj != null && root.has("outputs")) {
                int paletteSize = paletteObj.get("size").getAsInt();
                JsonObject outputs = root.getAsJsonObject("outputs");
                if (outputs.has("block_logits")) {
                    JsonArray bl = outputs.getAsJsonArray("block_logits");
                    if (bl.size() >= 2) {
                        JsonElement dim1 = bl.get(1);
                        if (dim1.isJsonPrimitive() && dim1.getAsJsonPrimitive().isString()) {
                            String s = dim1.getAsString();
                            if ("N_blocks".equalsIgnoreCase(s) || "n_blocks".equalsIgnoreCase(s)) {
                                bl.set(1, GSON.toJsonTree(paletteSize));
                            }
                        }
                    }
                }
            }
        } catch (Exception e) {
            // Don't fail on preprocessing; we'll surface a clearer error during validate()
            LOGGER.fine("Config preprocessing warning for " + jsonPath + ": " + e.getMessage());
        }

        // Now deserialize using snake_case mapping
        ModelConfig config = GSON.fromJson(root, ModelConfig.class);
        if (config == null) {
            throw new IOException("Failed to parse config: " + jsonPath);
        }
        config.validate();
        LOGGER.info("Loaded model config: " + config.modelName() + " from " + jsonPath);
        return config;
    }

    private static void renameKey(JsonObject obj, String from, String to) {
        if (obj.has(from) && !obj.has(to)) {
            JsonElement v = obj.remove(from);
            obj.add(to, v);
        }
    }
}
