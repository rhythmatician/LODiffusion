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
		
		// Register command
		CommandRegistrationCallback.EVENT.register((dispatcher, registryAccess, environment) -> {
			LodiffusionCommand.register(dispatcher);
		});
		
		LOGGER.info("[LODiffusion] Registered /lodiffusion command");
	}
}
