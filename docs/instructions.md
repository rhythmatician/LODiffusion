# Developer Instructions

Welcome to the AI-Diffusion Minecraft Mod project. Follow these steps using the **micro-commit strategy** to keep Copilot productive and PRs manageable.

## 🚀 Micro-Commit Workflow

### Before Starting Any Work
1. **Ensure clean state**: `git status` should show no uncommitted changes
2. **Get latest changes**: `git fetch && git checkout main && git pull`
3. **Create focused branch**: `git checkout -b test/add-single-method-test`
   - Use specific branch names: `test/add-chunk-lod-tests` not `test/improve-coverage`
   - Target ONE specific change per branch

### Development Cycle (15-20 minute sprints)
1. **Write one focused test** OR **implement minimal code** to pass existing test
2. **Commit immediately**: `git add . && git commit -m "test: add heightmap sampling test"`
3. **Push frequently**: `git push origin branch-name` for backup
4. **Repeat**: Keep commits small and focused

### Branch Completion
1. **Verify all tests pass**: `./gradlew test`
2. **Check coverage**: `./gradlew jacocoTestReport` (≥ 80% target)
3. **Final push**: `git push origin branch-name`
4. **Create PR**: Should be reviewable in < 10 minutes, max 200 lines changed

## 🔪 Test-First Development

1. **If tests aren't set up yet**, open `build.gradle`, add a `java` plugin, JUnit 5 dependencies, `useJUnitPlatform()`, and the JaCoCo plugin block.
2. **Write a single, focused test** in `src/test/java/...` (e.g., sampling the vanilla heightmap).
3. **Commit immediately**: `git add . && git commit -m "test: add vanilla heightmap sampling test"`

## 🛠️ Implementation

1. Open the corresponding class in `src/main/java/...`.
2. Place your cursor inside the new test's target method or add a `// TODO` comment stub.
3. **Invoke Copilot** to generate only the code needed to make the test pass.
4. Clean up generated code:
   - Remove any unrelated or unused code.
   - **For NBT/Anvil operations**: Use Querz/NBT library with proper error handling
   - **For biome parsing**: Implement version-aware logic (pre-1.18 vs 1.18+)
5. **Commit immediately**: `git add . && git commit -m "feat: implement heightmap sampling"`

## ✅ Verify & Refine

1. **Run local checks**: `./gradlew clean lint test jacocoTestReport` (matches CI pipeline).
2. Confirm your new test is green.
3. Check coverage: total coverage ≥ 80%.
4. If coverage dips, write additional tests rather than lowering the threshold.
5. **Commit any fixes**: `git add . && git commit -m "fix: resolve compilation error in heightmap test"`
6. **Troubleshooting**: If build fails, try `./gradlew build --refresh-dependencies` or `./gradlew cleanloom`

### CI Pipeline Structure
The CI runs three separate jobs for faster feedback:
- **Lint Job**: Fast code quality checks (`./gradlew lint`)
- **Test Job**: Unit tests + coverage (`./gradlew test jacocoTestReport`)
- **Build Job**: Final mod JAR build (only if lint + test pass)

### Gradle Troubleshooting Tips
If you see this:
```
Previous process has disowned the lock due to abrupt termination.
Found existing cache lock file (ACQUIRED_PREVIOUS_OWNER_DISOWNED), rebuilding loom cache.
```
Try the following steps:
```bash
rm -rf .gradle build                      # clean local project cache
rm -rf ~/.gradle/caches/fabric-loom      # (optional) clear Loom's global cache
./gradlew --stop                         # stop all Gradle daemons
./gradlew clean build                    # retry from a clean state
```
Avoid hard-killing builds (e.g., force-closing VS Code or Ctrl+C twice). Use `--no-daemon` in CI environments.

## 🧪 Phase 3 Implementation Details

### NBT Parsing Implementation Status
- **Library**: Using `io.github.querz:nbt:6.1` for robust NBT handling
- **Heightmap Extraction**: Implemented with MOTION_BLOCKING preference and WORLD_SURFACE fallback
- **Packed Data Decoding**: Custom decoder for 9-bit values packed in long arrays
- **Biome Compatibility**: Handles both pre-1.18 (simple int array) and 1.18+ (compound tag) formats
- **Error Recovery**: Graceful fallback to placeholder data for missing/corrupt chunks

### Build System Considerations
When working with the NBT implementation:
```bash
# If experiencing Gradle cache locks:
./gradlew --stop                          # Stop all daemons
rm -rf .gradle build                      # Clear local cache
./gradlew clean build --refresh-dependencies  # Clean rebuild

# For persistent Loom issues:
rm -rf ~/.gradle/caches/fabric-loom       # Clear Loom cache
```

### Testing Strategy for NBT Code
1. **Unit Tests**: Test individual NBT parsing methods with mock data
2. **Integration Tests**: Use real `.mca` files from test worlds
3. **Error Handling Tests**: Validate behavior with corrupt/missing data
4. **Performance Tests**: Benchmark parsing speed with large region files

## 📚 Update Documentation

1. **docs/PROJECT-OUTLINE.md**: mark the task complete (`- [x]`) or refine future steps if needed.
2. **Commit documentation changes**: `git add docs/ && git commit -m "docs: complete DiffusionModel.run integration"`

## 🔄 Reflection & Learning Cycle

After each completed task with all tests passing:

1. **Commit Working Changes**:
   - Commit implementation, tests, and documentation.
2. **Assess what worked**: Did Copilot guess well? What made it easier?
3. **Update Instructions**: Did you have to guide Copilot in a surprising way?
4. **Refactor**: Extract helpers, rename things, and prepare for the next cycle.

This reflection cycle ensures that knowledge gained from each task informs and improves future development cycles.

## 🌿 Branch & PR Discipline

1. Develop each micro-feature on its own focused branch (`test/add-xyz`, `feat/implement-abc`, `docs/update-def`).
2. **Keep PRs small**: max 200 lines changed, reviewable in < 10 minutes.
3. Push branch frequently during development for backup.
4. **Only merge** when:
   - All tests pass.
   - Coverage threshold is met (≥ 80%).
   - Documentation updates included where relevant.
5. **Enable auto-merge** for Copilot-reviewed PRs with no unresolved threads.
6. Use clear commit messages:
   - **`test:`** for new tests
   - **`feat:`** for implementation
   - **`fix:`** for bug fixes
   - **`docs:`** for documentation updates

## 👥 Supervisor Role
- Review every PR diff, focusing on API correctness and code quality.
- Guide Copilot with inline comments and clear method/test names.
- Pivot or refine the outline when higher-level priorities change.
- Approve and enable auto-merge on PRs with no unresolved threads.

## 🛠️ Technical Standards

### NBT Parsing & World Data
- **Library**: Use `io.github.querz:nbt:6.1` for all NBT operations
- **File Operations**: Always use try-with-resources for MCA file reading
- **Validation**: Check NBT tag existence before access (`tag.containsKey()`)
- **Heightmaps**: Extract MOTION_BLOCKING with fallback to WORLD_SURFACE
- **Biomes**: Handle both pre-1.18 and 1.18+ formats with version detection
- **Packed Data**: Use helper methods for decoding packed long arrays (9-bit values)

### Performance & Coordinate Systems
- **Chunk Math**: Use bitwise operations (`chunkX >> 5` instead of `/32`)
- **Memory**: Prefer primitive arrays over collections for heightmap data
- **Caching**: Implement memoization for expensive NBT parsing operations

### Error Handling & Logging
- **Specific Exceptions**: Catch IOException, DataFormatException separately
- **Fallback Values**: Provide sensible defaults for missing/corrupt chunk data
- **Detailed Logging**: Include chunk coordinates and file paths in error messages
- **Graceful Degradation**: Continue operation with placeholder data when possible

### Build System & Dependencies
- **Gradle Issues**: If cache locks occur, use `./gradlew --stop && rm -rf .gradle build`
- **Loom Cache**: Clear with `rm -rf ~/.gradle/caches/fabric-loom` if needed
- **Clean Builds**: Use `./gradlew clean build --refresh-dependencies` for dependency issues
- **Daemon Management**: Avoid force-killing builds; use `--no-daemon` in CI
