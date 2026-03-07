package com.rhythmatician.lodiffusion;

import com.rhythmatician.lodiffusion.voxy.LodGenerationService;
import com.rhythmatician.lodiffusion.voxy.VoxyCompat;

import net.fabricmc.api.ClientModInitializer;
import net.fabricmc.api.EnvType;
import net.fabricmc.api.Environment;
import net.fabricmc.fabric.api.client.event.lifecycle.v1.ClientTickEvents;
import net.fabricmc.fabric.api.client.networking.v1.ClientPlayConnectionEvents;

/**
 * Client-side entrypoint for LODiffusion.
 *
 * <p>Registers lifecycle events to start/stop the background LOD generation
 * service when the player joins/leaves a world.  The service runs on a
 * daemon thread and feeds ONNX-generated terrain into Voxy for distant
 * LOD rendering.
 */
@Environment(EnvType.CLIENT)
public class LodiffusionClient implements ClientModInitializer {

    private static final LodGenerationService LOD_SERVICE = new LodGenerationService();

    @Override
    public void onInitializeClient() {
        HelloTerrainMod.LOGGER.info("[LODiffusion] Client initializer starting");

        // --- World join: start LOD generation ---
        ClientPlayConnectionEvents.JOIN.register((handler, sender, client) -> {
            if (!VoxyCompat.isAvailable()) {
                HelloTerrainMod.LOGGER.warn("[LODiffusion] Voxy not available — LOD generation disabled");
                return;
            }
            if (!Config.useOnnxTerrain()) {
                HelloTerrainMod.LOGGER.info("[LODiffusion] ONNX terrain disabled in config");
                return;
            }

            // Delay service start slightly so the world has time to initialize
            client.execute(() -> {
                if (client.world != null) {
                    HelloTerrainMod.LOGGER.info("[LODiffusion] World joined — starting LOD generation service");
                    LOD_SERVICE.start(client.world);
                }
            });
        });

        // --- World leave: stop LOD generation ---
        ClientPlayConnectionEvents.DISCONNECT.register((handler, client) -> {
            HelloTerrainMod.LOGGER.info("[LODiffusion] Disconnected — stopping LOD generation service");
            LOD_SERVICE.stop();
        });

        // --- Client tick: update player position ---
        ClientTickEvents.END_CLIENT_TICK.register(client -> {
            if (LOD_SERVICE.isRunning() && client.player != null) {
                LOD_SERVICE.updatePlayerPosition(client.player.getBlockPos());
            }
        });

        HelloTerrainMod.LOGGER.info("[LODiffusion] Client initializer complete");
    }

    /** Get the active LOD generation service (for status commands etc). */
    public static LodGenerationService getLodService() {
        return LOD_SERVICE;
    }
}
