# Copilot Assistant Guide

## General Workflow
- Always follow TDD: write one focused JUnit test first, then implement just enough code to pass it.
- Before each feature, run `git status` to ensure a clean working tree.
- Use Java 17 syntax and Fabric API 1.21+ (`net.fabricmc.fabric.api.*`).
- Keep methods small; extract complex logic into helper classes (`DiffusionModel`, `LODQuery`, etc.).
- After each feature, mark it in `PROJECT-OUTLINE.md` with “- [x]” or update the upcoming tasks.
- Pin all dependency versions in `build.gradle`.

## Testing & CI
- Tests live under `src/test/java/...`; every new feature must include a test.
- Use JUnit 5 (`junit-jupiter-api`, `junit-jupiter-engine`) and Mockito (`mockito-core`).
- Code coverage target: **≥ 80%** on every commit. Enforced via `jacocoTestReport`.
- Run `./gradlew test jacocoTestReport lint` before opening a PR.
- Before opening a PR, Copilot must run:
  `./gradlew clean test jacocoTestReport lint`
  and only continue if all steps succeed.
- Track open issues using `CI-CHECKLIST.md`.

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

## PR & Branching Policy
- Use GitHub Flow:
  - One micro-feature per branch: `feature/<desc>`
  - PRs must include test coverage and pass CI
- Enable auto-merge if Copilot review leaves no unresolved threads.
- Copilot should approve and enable auto-merge on PRs it reviews, unless it opens threads requiring human input.
- Tag commits using prefixes:
  - `test:` for test additions
  - `feat:` for features
  - `docs:` for documentation
  - `fix:` for bugfixes
- Pull requests should update `PROJECT-OUTLINE.md` and `CI-CHECKLIST.md` if relevant.
- After passing tests, Copilot must `git add`, `git commit`, and include a descriptive message using `test:`, `feat:`, or `fix:` prefix.

## File Index
- `.github/copilot-instructions/anvil.md`: Anvil file format and NBT parsing
- `.github/copilot-instructions/chunk-extraction.md`: Chunk data extraction
- `.github/copilot-instructions/development.md`: General development practices
- `.github/copilot-instructions/distant-horizons-integration.md`: DH-specific logic and fallback patterns
- `instructions.md`: Developer instructions for Copilot usage
- `PROJECT-OUTLINE.md`: Project outline and task tracking
