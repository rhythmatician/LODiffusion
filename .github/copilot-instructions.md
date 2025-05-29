# Copilot Assistant Guide

General Workflow:
- Always follow TDD: write one focused JUnit test first, then implement just enough code to pass it.
- Use Java 17 syntax and Fabric 1.20+ API (`net.fabricmc.fabric.api.*`).
- Keep methods small; push heavy logic into helper classes (e.g., `DiffusionModel`).
- After each feature, update `PROJECT-OUTLINE.md` with an “- [x]” or refine upcoming tasks.
- Pin dependency versions in `build.gradle` to maintain consistency.

Testing & CI:
- Tests live under `src/test/java/...`; ensure each new behavior has a corresponding test.
- Use JUnit 5 (`junit-jupiter-api` & `junit-jupiter-engine`) and Mockito (`mockito-core`).
- Ensure Jacoco coverage ≥ 80% on every commit.

Shell Commands & HTTP:
- Whitelist auto-approved commands (no prompt):
  - `ls`, `git`, `grep`, `sed`, `awk`
  - `curl -X GET`, `curl --request GET`
- All other shell or HTTP methods (POST/PUT/DELETE) must prompt for approval.

Fabric & Mod Setup:
- When scaffolding, use the Fabric example mod as a base.
- In `build.gradle`, apply `java` & `jacoco` plugins and configure `test { useJUnitPlatform() }`.

Chunk Generation & Diffusion:
- In `DiffusionChunkGenerator.buildSurface(...)`, sample vanilla heightmap and biomes, then call `DiffusionModel.run(...)`.
- Enforce **Progressive Refinement**: each LOD pass must refine the previous level’s output, not recompute from scratch.

Distant Horizons Integration:
- Add dependency in Gradle:
  `modImplementation "com.terraformersmc:distant-horizons:<version>"`
- Import and call:
  ```java
  import com.terraformersmc.distanthorizons.api.LODManager;
  int lod = LODManager.getChunkLOD(player, chunk.getPos());
````

* Map `lod` to diffusion factors in a `switch` or `if` chain; write a `LODIntegrationTest` for each mapping.

Documentation & Branching:

* Use GitHub Flow: `feature/<desc>` branches, one microfeature per branch.
* Require CI & coverage checks before merging to `main`.
* Tag commits with `test:`, `feat:`, or `docs:` prefixes.

Repeat this cycle to keep Copilot focused, reliable, and aligned with our project goals.

