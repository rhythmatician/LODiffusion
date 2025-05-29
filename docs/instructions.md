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

## 🧪 Test-First Development
## 🧪 Test-First Development

1. **If tests aren't set up yet**, open `build.gradle`, add a `java` plugin, JUnit 5 dependencies, `useJUnitPlatform()`, and the JaCoCo plugin block.
2. **Write a single, focused test** in `src/test/java/...` (e.g., sampling the vanilla heightmap).
3. **Commit immediately**: `git add . && git commit -m "test: add vanilla heightmap sampling test"`

## 🔨 Implementation
## 🔨 Implementation

1. Open the corresponding class in `src/main/java/...`.
2. Place your cursor inside the new test's target method or add a `// TODO` comment stub.
3. **Invoke Copilot** to generate only the code needed to make the test pass.
4. **Review** the suggestion:
   - Ensure correct Fabric API usage and imports.
   - Validate edge-case handling and performance considerations.
   - Remove any unrelated or unused code.
   - **For NBT/Anvil operations**: Use Querz/NBT library with proper error handling
   - **For biome parsing**: Implement version-aware logic (pre-1.18 vs 1.18+)
5. **Commit immediately**: `git add . && git commit -m "feat: implement heightmap sampling"`

## ✅ Verify & Refine
## ✅ Verify & Refine

1. Run `./gradlew clean test jacocoTestReport`.
2. Confirm your new test is green.
3. Check coverage: total coverage ≥ 80%.
4. If coverage dips, write additional tests rather than lowering the threshold.
5. **Commit any fixes**: `git add . && git commit -m "fix: resolve compilation error in heightmap test"`
6. **Troubleshooting**: If build fails, try `./gradlew build --refresh-dependencies` or `./gradlew cleanloom`

## 📚 Update Documentation
## 📚 Update Documentation

1. **docs/PROJECT-OUTLINE.md**: mark the task complete (`- [x]`) or refine future steps if needed.
2. **Commit documentation changes**: `git add docs/ && git commit -m "docs: complete DiffusionModel.run integration"`

## 🔄 Reflection & Learning Cycle
After each completed task with all tests passing:

1. **Commit Working Changes**:
   - `git add .` to stage your changes
   - `git status` to review what files have been modified
   - `git diff` each file, to make sure no errors get committed
   - `git commit -m "feat: [task description]"`
   - Ensure the commit includes all working code and passing tests

2. **Reflect on the Process**:
   - What challenges were encountered during implementation?
   - Were there any API usage patterns that worked particularly well?
   - Did any edge cases emerge that should inform future tasks?
   - Were there any performance considerations or optimization opportunities?
   - What debugging techniques proved most effective?

3. **Update Learning Artifacts**:
   - Add new insights to the "Technical Standards" section below
   - Document any new dependency patterns or API usage discoveries
   - Note any build/test troubleshooting solutions that worked
   - Record any Fabric-specific gotchas or best practices learned

4. **Commit Learning Updates**:
   - Keep learning commits separate from implementation commits
   - `git status` to verify only documentation files are being committed
   - `git add instructions.md` (and any other documentation updates)
   - `git commit -m "docs: reflect on [task] - learned [key insight]"`
   - `git push origin`

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
## 👥 Supervisor Role

- Review every PR diff, focusing on API correctness and code quality.
- Guide Copilot with inline comments and clear method/test names.
- Pivot or refine the outline when higher-level priorities change.
- Approve and enable auto-merge on PRs with no unresolved threads.

## 🛠️ Technical Standards
- **NBT Parsing**: Always use try-with-resources for file operations and validate NBT tag existence
- **Coordinate Systems**: Use bitwise operations for performance (`chunkX >> 5` instead of `/32`)
- **Error Handling**: Catch specific exceptions (IOException, DataFormatException) with detailed logging
- **Version Compatibility**: Detect Minecraft version via DataVersion in level.dat when needed
- **Resource Management**: Explicit cleanup for DJL Predictors and MappedByteBuffer instances

Repeat this cycle for each new capability—keeping the loop tight ensures Copilot stays productive and aligned with project goals.
