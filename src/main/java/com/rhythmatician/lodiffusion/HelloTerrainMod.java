package com.rhythmatician.lodiffusion;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rhythmatician.lodiffusion.command.LodiffusionCommand;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;

public class HelloTerrainMod implements ModInitializer {
	public static final String MOD_ID = "lodiffusion";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		LOGGER.info("[LODiffusion] Mod initialized!");
		LOGGER.info("LODiffusion is ready to enhance terrain generation with AI-powered diffusion!");
		
		// Register command first to avoid any issues
		try {
			CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {
				LodiffusionCommand.register(dispatcher);
			});
			LOGGER.info("[LODiffusion] Registered /lodiffusion command");
		} catch (Exception e) {
			LOGGER.error("[LODiffusion] Failed to register command: " + e.getMessage(), e);
		}
		
		// Force early initialization of main OnnxTerrainGenerator to ensure bridge works
		LOGGER.info("[LODiffusion] About to initialize ONNX terrain generator...");
		try {
			LOGGER.info("[LODiffusion] Calling OnnxTerrainGenerator.getInstance()...");
			OnnxTerrainGenerator mainGenerator = OnnxTerrainGenerator.getInstance();
			LOGGER.info("[LODiffusion] getInstance() call completed, result: {}", mainGenerator != null ? "non-null" : "null");
			
			if (mainGenerator != null) {
				LOGGER.info("[LODiffusion] ✅ Main OnnxTerrainGenerator initialized successfully during mod init");
				LOGGER.info("[LODiffusion] Checking if main generator is available: {}", mainGenerator.isAvailable());
			} else {
				LOGGER.warn("[LODiffusion] ⚠️ Main OnnxTerrainGenerator getInstance() returned null");
			}
		} catch (Throwable e) {
			LOGGER.error("[LODiffusion] ❌ Failed to initialize main OnnxTerrainGenerator: " + e.getMessage(), e);
		}
		
		LOGGER.info("[LODiffusion] Mod initialization complete!");
	}
}
