package com.rhythmatician.lodiffusion.voxy;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;

import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class LodGenerationServiceModelDirResolutionTest {

    @TempDir
    Path tempDir;

    @Test
    void candidateModelDirs_includesRunAndParentChain() {
        Path projectRoot = tempDir.resolve("project");
        Path runDir = projectRoot.resolve("run");
        Path cwd = runDir;
        Path configured = Path.of("config", "lodiffusion");

        List<Path> candidates = LodGenerationService.candidateModelDirs(configured, cwd);

        assertEquals(runDir.resolve("config/lodiffusion").normalize(), candidates.get(0));
        assertTrue(candidates.contains(projectRoot.resolve("config/lodiffusion").normalize()));
    }

    @Test
    void findVoxyModelDir_prefersParentWhenRunDirMissingModels() throws Exception {
        Path projectRoot = tempDir.resolve("project");
        Path runDir = projectRoot.resolve("run");
        Path runConfig = runDir.resolve("config/lodiffusion");
        Path rootConfig = projectRoot.resolve("config/lodiffusion");
        Files.createDirectories(runConfig);
        Files.createDirectories(rootConfig);

        Files.write(rootConfig.resolve("voxy_l0.onnx"), new byte[] {1});

        Path resolved = LodGenerationService.findVoxyModelDir(Path.of("config", "lodiffusion"), runDir);
        assertEquals(rootConfig.normalize(), resolved);
    }

    @Test
    void findSparseModelDir_prefersParentWhenRunDirMissingModel() throws Exception {
        Path projectRoot = tempDir.resolve("project");
        Path runDir = projectRoot.resolve("run");
        Path runConfig = runDir.resolve("config/lodiffusion");
        Path rootConfig = projectRoot.resolve("config/lodiffusion");
        Files.createDirectories(runConfig);
        Files.createDirectories(rootConfig);

        Files.write(rootConfig.resolve("sparse_octree.onnx"), new byte[] {1});

        Path resolved = LodGenerationService.findSparseModelDir(Path.of("config", "lodiffusion"), runDir);
        assertEquals(rootConfig.normalize(), resolved);
    }

    @Test
    void findVoxyModelDir_findsModuleConfigFromWorkspaceRootCwd() throws Exception {
        Path workspaceRoot = tempDir.resolve("MC");
        Path moduleConfig = workspaceRoot.resolve("LODiffusion/config/lodiffusion");
        Files.createDirectories(moduleConfig);
        Files.write(moduleConfig.resolve("voxy_l0.onnx"), new byte[] {1});

        Path resolved = LodGenerationService.findVoxyModelDir(
                Path.of("config", "lodiffusion"),
                workspaceRoot);

        assertEquals(moduleConfig.normalize(), resolved);

        List<Path> candidates = LodGenerationService.candidateModelDirs(
                Path.of("config", "lodiffusion"),
                workspaceRoot);
        assertTrue(candidates.contains(moduleConfig.normalize()));
    }
}
