package com.rhythmatician.lodiffusion.command;

import com.mojang.brigadier.CommandDispatcher;
import com.mojang.brigadier.arguments.StringArgumentType;
import com.mojang.brigadier.context.CommandContext;
import com.rhythmatician.lodiffusion.Config;
import com.rhythmatician.lodiffusion.terrain.OnnxTerrainGenerator;
import com.rhythmatician.lodiffusion.terrain.adapter.AdapterRegistry;
import com.rhythmatician.lodiffusion.terrain.infer.ModelManager;
import com.rhythmatician.lodiffusion.util.DebugUtils;
import com.rhythmatician.lodiffusion.util.PerformanceMonitor;

import net.minecraft.server.command.CommandManager;
import net.minecraft.server.command.ServerCommandSource;
import net.minecraft.text.Text;

/**
 * Command interface for LODiffusion management and debugging.
 * Provides runtime control over ONNX terrain generation.
 */
public final class LodiffusionCommand {
    
    /**
     * Register the /lodiffusion command with the server.
     */
    public static void register(CommandDispatcher<ServerCommandSource> dispatcher) {
        dispatcher.register(CommandManager.literal("lodiffusion")
            .requires(source -> source.hasPermissionLevel(2)) // Requires OP permissions
            
            // Status subcommand
            .then(CommandManager.literal("status")
                .executes(context -> executeStatus(context)))
            
            // Toggle ONNX terrain generation
            .then(CommandManager.literal("toggle")
                .executes(context -> executeToggle(context)))
            
            // Change adapter
            .then(CommandManager.literal("adapter")
                .then(CommandManager.argument("adapter_name", StringArgumentType.string())
                    .suggests((context, builder) -> {
                        for (String adapter : AdapterRegistry.getAvailableAdapters()) {
                            builder.suggest(adapter);
                        }
                        return builder.buildFuture();
                    })
                    .executes(context -> executeSetAdapter(context))))
            
            // Performance report
            .then(CommandManager.literal("performance")
                .executes(context -> executePerformance(context)))
            
            // Reset metrics
            .then(CommandManager.literal("reset")
                .executes(context -> executeReset(context)))
            
            // System debug report
            .then(CommandManager.literal("debug")
                .executes(context -> executeDebug(context)))
            
            // Reload model
            .then(CommandManager.literal("reload")
                .executes(context -> executeReload(context)))
        );
    }
    
    private static int executeStatus(CommandContext<ServerCommandSource> context) {
        ServerCommandSource source = context.getSource();
        
        StringBuilder status = new StringBuilder();
        status.append("§6=== LODiffusion Status ===§r\n");
        status.append("§7ONNX Terrain: §").append(Config.useOnnxTerrain() ? "aEnabled" : "cDisabled").append("§r\n");
        status.append("§7Current Adapter: §f").append(Config.adapter()).append("§r\n");
        status.append("§7Model Available: §").append(ModelManager.isAvailable() ? "aYes" : "cNo").append("§r\n");
        status.append("§7System Ready: §").append(OnnxTerrainGenerator.isReady() ? "aYes" : "cNo").append("§r\n");
        
        long chunksGenerated = PerformanceMonitor.getCounter(PerformanceMonitor.CHUNKS_GENERATED);
        long onnxInferences = PerformanceMonitor.getCounter(PerformanceMonitor.ONNX_INFERENCES);
        long fallbackUses = PerformanceMonitor.getCounter(PerformanceMonitor.FALLBACK_USES);
        
        status.append("§7Chunks Generated: §f").append(chunksGenerated).append("§r\n");
        status.append("§7ONNX Inferences: §f").append(onnxInferences).append("§r\n");
        status.append("§7Fallback Uses: §f").append(fallbackUses).append("§r\n");
        
        if (chunksGenerated > 0) {
            double onnxRate = (onnxInferences * 100.0) / chunksGenerated;
            status.append("§7ONNX Success Rate: §f").append(String.format("%.1f%%", onnxRate)).append("§r");
        }
        
        source.sendFeedback(() -> Text.literal(status.toString()), false);
        return 1;
    }
    
    private static int executeToggle(CommandContext<ServerCommandSource> context) {
        ServerCommandSource source = context.getSource();
        
        boolean newState = !Config.useOnnxTerrain();
        Config.setUseOnnxTerrain(newState);
        
        String message = String.format("§6LODiffusion ONNX terrain generation §%s%s§6.§r", 
            newState ? "a" : "c", newState ? "enabled" : "disabled");
        source.sendFeedback(() -> Text.literal(message), true);
        
        return 1;
    }
    
    private static int executeSetAdapter(CommandContext<ServerCommandSource> context) {
        ServerCommandSource source = context.getSource();
        String adapterName = StringArgumentType.getString(context, "adapter_name");
        
        if (!AdapterRegistry.hasAdapter(adapterName)) {
            String available = String.join(", ", AdapterRegistry.getAvailableAdapters());
            source.sendFeedback(() -> Text.literal(
                "§cUnknown adapter: " + adapterName + "\n§7Available adapters: " + available + "§r"), false);
            return 0;
        }
        
        Config.setAdapter(adapterName);
        source.sendFeedback(() -> Text.literal(
            "§6Changed adapter to: §f" + adapterName + "§r"), true);
        
        return 1;
    }
    
    private static int executePerformance(CommandContext<ServerCommandSource> context) {
        ServerCommandSource source = context.getSource();
        
        String report = PerformanceMonitor.getPerformanceReport();
        String[] lines = report.split("\n");
        
        for (String line : lines) {
            if (!line.trim().isEmpty()) {
                source.sendFeedback(() -> Text.literal("§7" + line + "§r"), false);
            }
        }
        
        return 1;
    }
    
    private static int executeReset(CommandContext<ServerCommandSource> context) {
        ServerCommandSource source = context.getSource();
        
        PerformanceMonitor.reset();
        source.sendFeedback(() -> Text.literal("§6Reset all performance metrics.§r"), true);
        
        return 1;
    }
    
    private static int executeDebug(CommandContext<ServerCommandSource> context) {
        ServerCommandSource source = context.getSource();
        
        String report = DebugUtils.createSystemReport();
        String[] lines = report.split("\n");
        
        // Send report in chunks to avoid chat spam
        int chunkSize = 10;
        for (int i = 0; i < lines.length; i += chunkSize) {
            StringBuilder chunk = new StringBuilder();
            for (int j = i; j < Math.min(i + chunkSize, lines.length); j++) {
                if (!lines[j].trim().isEmpty()) {
                    chunk.append("§7").append(lines[j]).append("§r\n");
                }
            }
            
            if (chunk.length() > 0) {
                final String finalChunk = chunk.toString();
                source.sendFeedback(() -> Text.literal(finalChunk), false);
                
                // Small delay between chunks to prevent flooding
                try {
                    Thread.sleep(100);
                } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                }
            }
        }
        
        return 1;
    }
    
    private static int executeReload(CommandContext<ServerCommandSource> context) {
        ServerCommandSource source = context.getSource();
        
        try {
            ModelManager.close();
            source.sendFeedback(() -> Text.literal("§6Closed existing model.§r"), false);
            
            // Try to reload
            if (ModelManager.isAvailable()) {
                source.sendFeedback(() -> Text.literal("§aModel reloaded successfully.§r"), true);
            } else {
                source.sendFeedback(() -> Text.literal("§cFailed to reload model.§r"), true);
            }
            
        } catch (Exception e) {
            source.sendFeedback(() -> Text.literal("§cError reloading model: " + e.getMessage() + "§r"), true);
            return 0;
        }
        
        return 1;
    }
    
    // Prevent instantiation
    private LodiffusionCommand() {}
}
