package com.rhythmatician.lodiffusion;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.rhythmatician.lodiffusion.command.LodiffusionCommand;
import com.rhythmatician.lodiffusion.terrain.OnnxTerrainGenerator;
import com.rhythmatician.lodiffusion.voxy.VoxyCompat;

import net.fabricmc.api.ModInitializer;
import net.fabricmc.fabric.api.command.v2.CommandRegistrationCallback;

public class HelloTerrainMod implements ModInitializer {
	public static final String MOD_ID = "lodiffusion";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		LOGGER.info("[LODiffusion] Mod initialized!");

		// Register /lodiffusion command
		try {
			CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {
				LodiffusionCommand.register(dispatcher);
			});
			LOGGER.info("[LODiffusion] Registered /lodiffusion command");
		} catch (Exception e) {
			LOGGER.error("[LODiffusion] Failed to register command: {}", e.getMessage(), e);
		}

		// Detect companion mods
		LOGGER.info("[LODiffusion] {}", ModDetection.getLODStrategyInfo());

		if (VoxyCompat.isAvailable()) {
			LOGGER.info("[LODiffusion] Voxy reflection bindings OK — LOD injection path available");
		}

		// Check if model files are present (don't load yet — lazy on first chunk)
		if (OnnxTerrainGenerator.isReady()) {
			LOGGER.info("[LODiffusion] ONNX model + config found at {}", Config.modelPath());
		} else {
			LOGGER.warn("[LODiffusion] Model files not found at {} — terrain generation will fail until model is placed", Config.modelPath());
		}

		LOGGER.info("[LODiffusion] Mod initialization complete!");
	}
}
