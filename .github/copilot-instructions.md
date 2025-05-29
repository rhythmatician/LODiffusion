# Copilot Assistant Guide

## General Workflow
- Always follow TDD: write one focused JUnit test first, then implement just enough code to pass it.
- Before each feature, run `git status` to ensure a clean working tree.
- Use Java 17 syntax and Fabric API 1.21+ (`net.fabricmc.fabric.api.*`).
- Keep methods small; extract complex logic into helper classes (`DiffusionModel`, `LODQuery`, etc.).
- After each feature, mark it in `docs/PROJECT-OUTLINE.md` with “- [x]” or update the upcoming tasks.
- Pin all dependency versions in `build.gradle`.

## Testing & CI
- Tests live under `src/test/java/...`; every new feature must include a test.
- Use JUnit 5 (`junit-jupiter-api`, `junit-jupiter-engine`) and Mockito (`mockito-core`).
- Code coverage target: **≥ 80%** on every commit. Enforced via `jacocoTestReport`.
- Run `./gradlew.bat test jacocoTestReport lint` before opening a PR (use `.bat` extension for Windows).
- Before opening a PR, Copilot must run:
  `./gradlew.bat clean test jacocoTestReport lint`
  and only continue if all steps succeed.
- Track open issues using `docs/CI-CHECKLIST.md`.
- **Git Bash compatibility:** When working with paths, use Unix-style forward slashes. Change directory using `cd /c/Users/...` instead of `cd "c:\Users\..."`.

## Shell Commands & HTTP
- Whitelist auto-approved commands (no prompt):
  - `ls`, `git`, `grep`, `sed`, `awk`
  - `curl -X GET`, `curl --request GET`
- Prompt before using `POST`, `PUT`, `DELETE`, or file-modifying shell commands.

## Fabric & Mod Setup
- Scaffold mods using the Fabric example mod.
- Ensure `build.gradle` includes:
  - `java` and `jacoco` plugins
  - `test { useJUnitPlatform() }`
  - `compileOnly` or `modImplementation` as appropriate for integration dependencies

## Chunk Generation & Diffusion
- In `DiffusionChunkGenerator.buildSurface(...)`, sample vanilla heightmap and biomes, then call `DiffusionModel.run(...)`.
- Progressive LOD enforcement: each LOD refinement **must** operate on the previous level's output, not from scratch.
- Multi-channel support is in progress; tests should guide changes to `DiffusionModel`.

## Distant Horizons Integration
- Use **runtime detection** for DH integration unless hard-dependency is absolutely required.
- DH dependency (`com.seibel.distanthorizons:distant-horizons-api:4.0.0`) is marked as `compileOnly`.
- Wrap all calls in `ModDetection.isDistantHorizonsLoaded()` checks or use reflection fallback.
- Use `LODManager.getChunkLOD(player, chunk.getPos())` to determine LOD level.
- Implement and test mappings in `LODManagerCompat` and `DistantHorizonsCompat`.

## Git Workflow & Micro-Commit Strategy
**CRITICAL**: Never create massive PRs again. Use micro-commits for all development.

### Branch Management Workflow
1. **Before starting new work**:
   ```bash
   git fetch  # Check if previous PRs have merged
   git checkout main
   git pull
   ```

2. **Clean up completed branches**:
   ```bash
   git branch -d feature/old-branch-name  # Delete local branches
   ```

3. **Create focused feature branches**:
   ```bash
   git checkout -b test/add-single-method-test
   git checkout -b fix/compilation-error-line-45
   git checkout -b docs/update-coverage-metrics
   ```

### Micro-Commit Strategy
- **Commit every 15-20 minutes**: Even if feature isn't complete
- **One logical change per commit**: 
  - Add 1-2 test methods → commit
  - Fix one compilation issue → commit
  - Update one documentation section → commit
- **Push frequently**: Backup work and enable smaller PRs
- **Branch names should be specific**: `test/add-chunk-lod-tests` not `test/improve-coverage`

### PR & Branching Policy
- Use GitHub Flow with **micro-features**:
  - Each branch targets ONE specific change
  - PRs should be reviewable in < 10 minutes
  - Maximum 200 lines of changes per PR
- **Auto-merge enabled** for docs and small PRs when:
  - ✅ Only `docs/`, `*.md`, or `.github/workflows/*.yml` files changed
  - ✅ < 200 lines of code changed
  - ✅ All CI checks passing (lint, test, build)
  - ✅ No open review threads
- Copilot should approve and enable auto-merge on PRs it reviews, unless it opens threads requiring human input.
- Tag commits using prefixes:
  - `test:` for test additions
  - `feat:` for features
  - `docs:` for documentation
  - `fix:` for bugfixes
- Pull requests should update `docs/PROJECT-OUTLINE.md` and `docs/CI-CHECKLIST.md` if relevant.
- After passing tests, Copilot must `git add`, `git commit`, and include a descriptive message using `test:`, `feat:`, or `fix:` prefix.

## File Index
- `.github/copilot-instructions/anvil.md`: Anvil file format and NBT parsing
- `.github/copilot-instructions/chunk-extraction.md`: Chunk data extraction
- `.github/copilot-instructions/development.md`: General development practices
- `.github/copilot-instructions/distant-horizons-integration.md`: DH-specific logic and fallback patterns
- `docs/instructions.md`: Developer instructions for Copilot usage
- `docs/PROJECT-OUTLINE.md`: Project outline and task tracking
- `docs/EXAMPLE-WORLD-USAGE.md`: Example world data usage

## Coplilot's Journals
> It's good to review these, to avoid repeating the same mistakes:
- `PHASE-1-REFLECTION.md`: Reflection on Phase 1 development
