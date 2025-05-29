package com.rhythmatician.lodiffusion;

import net.fabricmc.api.ModInitializer;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

public class HelloTerrainMod implements ModInitializer {
	public static final String MOD_ID = "lodiffusion";
	public static final Logger LOGGER = LoggerFactory.getLogger(MOD_ID);

	@Override
	public void onInitialize() {
		LOGGER.info("[HelloTerrain] Mod initialized!");
		LOGGER.info("LODiffusion is ready to enhance terrain generation with AI-powered diffusion!");
	}
}
