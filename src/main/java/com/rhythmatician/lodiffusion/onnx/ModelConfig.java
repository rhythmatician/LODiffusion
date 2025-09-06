package com.rhythmatician.lodiffusion.onnx;

import java.util.Map;

/**
 * Model configuration loaded from model_config.json for each progressive LOD model.
 * Defines input/output shapes, normalization parameters, and block palette information.
 */
public record ModelConfig(
    String modelName,
    String version,
    Map<String, int[]> inputs,
    Map<String, int[]> optionalInputs,
    Map<String, int[]> outputs,
    NormalizationConfig normalization,
    BlockPalette blockPalette
) {
    
    /**
     * Normalization configuration for different feature types.
     */
    public record NormalizationConfig(
        HeightNormalization heights,
        RouterNormalization router6,
        BiomeNormalization biome,
        CoordNormalization coords
    ) {}
    
    /**
     * Height normalization using world bounds (MinMax scaling).
     */
    public record HeightNormalization(
        String type,  // "minmax"
        int bottomY,  // World bottom Y
        int height    // World height range
    ) {
        public float normalize(int rawHeight) {
            return (float)(rawHeight - bottomY) / height;
        }
        
        public int denormalize(float normalized) {
            return Math.round(normalized * height + bottomY);
        }
    }
    
    /**
     * Router field normalization using per-channel statistics (Z-score).
     */
    public record RouterNormalization(
        String type,    // "zscore"
        float[] mean,   // Per-channel mean [6 channels]
        float[] std     // Per-channel std [6 channels]
    ) {
        public float normalize(float rawValue, int channel) {
            if (channel < 0 || channel >= mean.length) {
                throw new IllegalArgumentException("Invalid channel: " + channel);
            }
            return (rawValue - mean[channel]) / std[channel];
        }
        
        public float denormalize(float normalized, int channel) {
            if (channel < 0 || channel >= mean.length) {
                throw new IllegalArgumentException("Invalid channel: " + channel);
            }
            return normalized * std[channel] + mean[channel];
        }
    }
    
    /**
     * Biome feature normalization (mixed types).
     */
    public record BiomeNormalization(
        String type  // "mixed" - temp continuous, flags binary
    ) {}
    
    /**
     * Coordinate normalization using tanh scaling.
     */
    public record CoordNormalization(
        String type,     // "tanh"
        float scale      // Scale factor for tanh(coord/scale)
    ) {
        public float normalize(int rawCoord) {
            return (float) Math.tanh(rawCoord / scale);
        }
        
        public int denormalize(float normalized) {
            // Approximate inverse: scale * atanh(normalized)
            double atanh = 0.5 * Math.log((1 + normalized) / (1 - normalized));
            return Math.round((float)(scale * atanh));
        }
    }
    
    /**
     * Block palette information for the model.
     */
    public record BlockPalette(
        int size,               // Number of block types (N_blocks)
        String mapping          // Reference to block mapping file
    ) {}
    
    /**
     * Get input shape for a specific tensor name.
     */
    public int[] getInputShape(String tensorName) {
        int[] shape = inputs.get(tensorName);
        if (shape == null) {
            shape = optionalInputs.get(tensorName);
        }
        if (shape == null) {
            throw new IllegalArgumentException("Unknown input tensor: " + tensorName);
        }
        return shape.clone();
    }
    
    /**
     * Get output shape for a specific tensor name.
     */
    public int[] getOutputShape(String tensorName) {
        int[] shape = outputs.get(tensorName);
        if (shape == null) {
            throw new IllegalArgumentException("Unknown output tensor: " + tensorName);
        }
        return shape.clone();
    }
    
    /**
     * Check if an input is optional.
     */
    public boolean isOptionalInput(String tensorName) {
        return optionalInputs.containsKey(tensorName);
    }
    
    /**
     * Get the output resolution (assumes cubic output).
     */
    public int getOutputResolution() {
        int[] blockLogitsShape = outputs.get("block_logits");
        if (blockLogitsShape == null || blockLogitsShape.length != 5) {
            throw new IllegalStateException("Invalid block_logits shape");
        }
        // Shape is [1, N_blocks, X, Y, Z] - assume X==Y==Z
        return blockLogitsShape[2];
    }
    
    /**
     * Get the parent input resolution (assumes cubic input).
     */
    public int getParentResolution() {
        int[] parentShape = inputs.get("x_parent_prev");
        if (parentShape == null || parentShape.length != 5) {
            throw new IllegalStateException("Invalid x_parent_prev shape");
        }
        // Shape is [1, 1, X, Y, Z] - assume X==Y==Z
        return parentShape[2];
    }
    
    /**
     * Validate configuration consistency.
     */
    public void validate() {
        // Check required inputs
        String[] requiredInputs = {
            "x_parent_prev", "x_height_planes", "x_biome_quart", 
            "x_router6", "x_chunk_pos", "x_lod"
        };
        
        for (String required : requiredInputs) {
            if (!inputs.containsKey(required)) {
                throw new IllegalStateException("Missing required input: " + required);
            }
        }
        
        // Check required outputs
        String[] requiredOutputs = {"block_logits", "air_mask"};
        for (String required : requiredOutputs) {
            if (!outputs.containsKey(required)) {
                throw new IllegalStateException("Missing required output: " + required);
            }
        }
        
        // Validate router normalization has 6 channels
        if (normalization.router6.mean.length != 6 || normalization.router6.std.length != 6) {
            throw new IllegalStateException("Router normalization must have 6 channels");
        }
        
        // Validate block palette size matches output
        int[] blockLogitsShape = outputs.get("block_logits");
        if (blockLogitsShape[1] != blockPalette.size) {
            throw new IllegalStateException("Block palette size mismatch with output shape");
        }
    }
}
