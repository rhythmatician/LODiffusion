package com.rhythmatician.lodiffusion;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import net.fabricmc.api.ModInitializer;

public class HelloTerrainMod implements ModInitializer {
	public static final String MOD_ID = "lodiffusion";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		LOGGER.info("[LODiffusion] Mod initialized!");
		LOGGER.info("LODiffusion is ready to enhance terrain generation with AI-powered diffusion!");
		
		// Test ONNX integration on startup
		testOnnxIntegration();
	}
	
	private void testOnnxIntegration() {
		try {
			LOGGER.info("[LODiffusion] Testing ONNX terrain generator...");
			
			try (OnnxTerrainGenerator generator = new OnnxTerrainGenerator()) {
				if (generator.isAvailable()) {
					LOGGER.info("[LODiffusion] ✅ ONNX terrain generator loaded successfully!");
					LOGGER.info("[LODiffusion] 🎮 Ready for AI-powered terrain generation!");
				} else {
					LOGGER.info("[LODiffusion] ⚠️ ONNX model not available, using enhanced fallback");
				}
			}
			
			// Test DiffusionModel integration  
			new DiffusionModel(); // Test instantiation
			LOGGER.info("[LODiffusion] ✅ DiffusionModel initialized successfully!");
			
		} catch (Exception e) {
			LOGGER.error("[LODiffusion] ❌ Error during ONNX testing: " + e.getMessage());
		}
	}
}
