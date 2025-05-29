# Developer Instructions

Welcome to the AI-Diffusion Minecraft Mod project. Follow these steps on every microfeature to keep Copilot on track:

## 1. Test-First Workflow
1. **If tests aren't set up yet**, open `build.gradle`, add a `java` plugin, JUnit 5 dependencies, `useJUnitPlatform()`, and the JaCoCo plugin block.
2. **Write a single, focused test** in `src/test/java/...` (e.g., sampling the vanilla heightmap).
3. **Commit and push** immediately—CI should fail because the test is unimplemented.

## 2. Implementation
1. Open the corresponding class in `src/main/java/...`.
2. Place your cursor inside the new test's target method or add a `// TODO` comment stub.
3. **Invoke Copilot** to generate only the code needed to make the test pass.
4. **Review** the suggestion:
   - Ensure correct Fabric API usage and imports.
   - Validate edge-case handling and performance considerations.
   - Remove any unrelated or unused code.
   - **For NBT/Anvil operations**: Use Querz/NBT library with proper error handling
   - **For biome parsing**: Implement version-aware logic (pre-1.18 vs 1.18+)

## 3. Verify & Refine
1. Run `./gradlew clean test jacocoTestReport`.
2. Confirm your new test is green.
3. Check coverage: total coverage ≥ 80%.
4. If coverage dips, write additional tests rather than lowering the threshold.
5. **Troubleshooting**: If build fails, try `./gradlew build --refresh-dependencies` or `./gradlew cleanloom`

## 4. Update Documentation
1. **PROJECT-OUTLINE.md**: mark the task complete (`- [x]`) or refine future steps if needed.
2. Commit any outline changes with a **`docs:`** prefix (e.g., `docs: complete DiffusionModel.run integration`).

## 5. Branch & PR Discipline
1. Develop each microfeature on its own branch (`feature/xxx`).
2. Push branch, open a PR, and let CI run.
3. **Only merge** when:
   - All tests pass.
   - Coverage threshold is met.
   - Outline updates are included where relevant.
4. Use clear commit messages:
   - **`test:`** for new tests
   - **`feat:`** for implementation
   - **`docs:`** for outline or instruction updates

## 6. Supervisor Role
- Review every PR diff, focusing on API correctness and code quality.
- Guide Copilot with inline comments and clear method/test names.
- Pivot or refine the outline when higher-level priorities change.

## 7. Technical Standards
- **NBT Parsing**: Always use try-with-resources for file operations and validate NBT tag existence
- **Coordinate Systems**: Use bitwise operations for performance (`chunkX >> 5` instead of `/32`)
- **Error Handling**: Catch specific exceptions (IOException, DataFormatException) with detailed logging
- **Version Compatibility**: Detect Minecraft version via DataVersion in level.dat when needed
- **Resource Management**: Explicit cleanup for DJL Predictors and MappedByteBuffer instances

Repeat this cycle for each new capability—keeping the loop tight ensures Copilot stays productive and aligned with project goals.
