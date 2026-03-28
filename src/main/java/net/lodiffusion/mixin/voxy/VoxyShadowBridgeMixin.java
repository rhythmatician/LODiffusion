package net.lodiffusion.mixin.voxy;

import com.rhythmatician.lodiffusion.HelloTerrainMod;
import net.lodiffusion.shadow.VoxyRequestDecoder;
import net.lodiffusion.shadow.ShadowRouterJobQueue;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;
import org.lwjgl.system.MemoryUtil;

/**
 * Fabric Mixin for VoxyShadowBridge.
 * 
 * Intercepts HierarchicalOcclusionTraverser.forwardDownloadResult() to extract
 * missing terrain requests from Voxy's GPU traversal and enqueue them for
 * demand-driven generation via LODiffusion's TerrainComputeDispatcher.
 * 
 * Target method signature:
 *   private void forwardDownloadResult(long ptr, long size)
 * 
 * Interception point: just before nodeManager.submitRequestBatch() is called,
 * after requests have been downloaded and parsed.
 */
@Mixin(targets = "me.cortex.voxy.client.core.rendering.hierachical.HierarchicalOcclusionTraverser")
public class VoxyShadowBridgeMixin {

    private static long lastBridgeLogMs;
    private static int bridgeBatchCount;
    private static int bridgeRequestCount;
    
    /**
     * Intercept the request batch callback to extract and enqueue requests for generation.
     * 
     * This injection runs at the beginning of forwardDownloadResult, giving us access to:
     * - ptr: native memory buffer containing [count(4)] + [request[0](8)] + ... 
     * - size: total buffer size in bytes
     * 
     * The buffer layout is:
     *   Offset 0–3: count (number of requests)
     *   Offset 4–7: padding (reserved)
     *   Offset 8+:  request array (each request is 8 bytes, uvec2)
     */
    @Inject(
        method = "forwardDownloadResult",
        at = @At("HEAD"),
        cancellable = false
    )
    private void interceptRequests(long ptr, long size, CallbackInfo ci) {
        try {
            if (ptr == 0 || size < 8) {
                return;  // Invalid buffer
            }
            
            // Read request count (first 4 bytes, little-endian int)
            int count = MemoryUtil.memGetInt(ptr);
            
            if (count <= 0 || count > 10000) {  // Sanity check
                return;
            }
            
            // Validate buffer has enough space
            long expectedSize = 8L + (long) count * 8L;
            if (size < expectedSize) {
                count = (int) ((size - 8) / 8);
                if (count <= 0) {
                    return;
                }
            }
            
            // Decode all requests from the buffer
            VoxyRequestDecoder.VoxyNodeRequest[] requests = 
                new VoxyRequestDecoder.VoxyNodeRequest[count];
            
            long requestPtr = ptr + 8;  // Skip count header
            for (int i = 0; i < count; i++) {
                try {
                    requests[i] = VoxyRequestDecoder.decode(requestPtr, i * 8);
                } catch (Exception e) {
                    // Log but don't crash on individual request decode errors
                    System.err.println("[LODiffusion] Error decoding request " + i + ": " + e);
                }
            }
            
            // Enqueue all valid requests to ShadowRouterJobQueue
            // This happens before Voxy's normal nodeManager.submitRequestBatch(),
            // so LODiffusion gets first crack at generating missing terrain
            ShadowRouterJobQueue.enqueueBatch(requests);
            logBridgeProgress(count);
            
        } catch (Exception e) {
            // Fail gracefully: don't crash Voxy if bridge code has issues
            System.err.println("[LODiffusion] VoxyShadowBridge error: " + e);
            e.printStackTrace();
        }
    }

    private static void logBridgeProgress(int requestCount) {
        bridgeBatchCount++;
        bridgeRequestCount += requestCount;

        long now = System.currentTimeMillis();
        if (bridgeBatchCount == 1 || now - lastBridgeLogMs >= 5000L) {
            HelloTerrainMod.LOGGER.info(
                    "[LodGen][Bridge] batches={} requests={} queued={} inFlight={}",
                    bridgeBatchCount,
                    bridgeRequestCount,
                    ShadowRouterJobQueue.size(),
                    ShadowRouterJobQueue.inFlightSize());
            lastBridgeLogMs = now;
        }
    }
}
