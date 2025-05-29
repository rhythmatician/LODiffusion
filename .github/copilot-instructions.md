# Copilot Assistant Instructions for LODiffusion

## 🔁 Development Workflow

Copilot must adhere to the **test-first, micro-commit strategy**:

### Before Each Feature
```bash
git fetch && git checkout main && git pull
````

* Ensure `git status` shows a clean working tree
* Create a focused branch with a clear prefix:

  * `test/add-xyz-test`
  * `feat/implement-abc`
  * `fix/resolve-def`
  * `docs/update-ghi`

```bash
git checkout -b test/add-xyz-test
```

### Development Cycle (Every 15–20 minutes)

1. **Write one small test** → commit (`test:`)
2. **Implement only enough to pass** → commit (`feat:`)
3. **Fix issues, refactor, or cleanup** → commit (`fix:` or `refactor:`)
4. **Push frequently** to enable CI and backups

```bash
git add .
git commit -m "test: add vanilla heightmap sampling"
git push origin your-branch-name
```

### Finalize Branch

1. Run:

   ```bash
   ./gradlew clean lint test jacocoTestReport build
   ```

   *Note: `lint` must pass before `test` or `build`.*
2. PR must:

   * Be under 200 LOC
   * Be reviewable in under 10 minutes
   * Pass all CI stages
   * Have no unresolved Copilot review threads

---

## 🔬 Testing & CI Discipline

### Test Rules

* Tests may live in:

  * `src/test/java/com/...` — core unit and integration tests
  * `src/test/java/data/` — synthetic dataset tests (e.g., `BiomeSamplingTest`)
  * `src/test/java/benchmark/` — performance and inference benchmarks
* Use **JUnit 5** and **Mockito**
* Target **80%+ code coverage per commit**
* Use tags for clarity:

  * `@Tag("ci")` — regular CI tests
  * `@Tag("inference")` — DJL/ONNX integration
  * `@Tag("benchmark")` — long-running benchmarks (excluded from default CI)

### CI Jobs

Each commit/PR runs:

1. **Lint**: `./gradlew lint` — must pass first
2. **Test + Coverage**: `./gradlew test jacocoTestReport`
   *Runs all `src/test/java/**` unless `@Tag("benchmark")` is excluded by config*
3. **Build Mod**: `./gradlew build` (only if lint + test pass)

Local equivalent:

```bash
./gradlew clean lint test jacocoTestReport build
```

---

## 🧠 Mod Responsibilities

### Chunk Generation & Diffusion
- In `DiffusionChunkGenerator.buildSurface(...)`:
  - Sample vanilla heightmap + biomes
  - Call `DiffusionModel.run(...)`
- **LOD chaining is required**: each refinement builds on the prior LOD
- Stubbed multi-channel logic must be test-guided and forward-compatible

### Distant Horizons Integration
- Runtime detection only (via `ModDetection.isDistantHorizonsLoaded()`)
- API dependency is `compileOnly`
- Use `LODManager.getChunkLOD(...)` for LOD level detection
- Implement fallback wrappers in `LODManagerCompat`, `DistantHorizonsCompat`

---

## 🧪 Implementation Patterns

### Java Conventions
- Java 17 required
- Fabric API 1.21+
- Use bitwise ops for chunk math (`chunkX >> 5`)
- Wrap file/NBT IO in try-with-resources
- Isolate logic: `DiffusionModel`, `ChunkSampler`, `LODQuery`, etc.
- Ensure `build.gradle` includes:
  - `java`, `jacoco` plugins
  - `test { useJUnitPlatform() }`
  - All dependency versions pinned

### Error Handling
- Catch only specific exceptions
- Log useful info for NBT/data failures

---

## 🧵 Git Branching & PR Discipline

### Micro-Commit Strategy
- Commit every 15–20 minutes
- One logical change per commit:
  - Add test → `test:`
  - Implement → `feat:`
  - Fix → `fix:`
  - Doc → `docs:`

### PR Requirements
- PRs must:
  - Contain only one logical change
  - Touch <200 LOC
  - Be reviewable in <10 minutes
  - Be auto-mergeable if:
    - ✅ Only `docs/`, `*.md`, `.github/` files changed
    - ✅ All CI checks pass
    - ✅ No Copilot threads open

### Commit Prefixes
- `test:` - New or updated tests
- `feat:` - New feature implementation
- `fix:` - Bug fix
- `docs:` - Markdown or outline update

---

## ☁️ Safe Shell Access

### Auto-approved Shell Commands
```bash
ls, git, grep, sed, awk,
curl -X GET, curl --request GET
```

### Prompt First (Copilot Must Ask)
- All POST/PUT/DELETE
- Any command modifying files or system state

---

## 🗂️ File Index
- `.github/copilot-instructions/anvil.md` — Anvil + NBT parsing
- `.github/copilot-instructions/chunk-extraction.md` — Chunk IO logic
- `.github/copilot-instructions/development.md` — Misc best practices
- `.github/copilot-instructions/distant-horizons-integration.md` — DH APIs + fallback
- `docs/CI-CHECKLIST.md` — Copilot’s own PR checklist
- `docs/PROJECT-OUTLINE.md` — Full project plan
- `docs/instructions.md` — Developer instructions

## Coplilot's Journals
- `PHASE-1-REFLECTION.md` — Copilot's journal of mistakes + learnings
- `docs\COVERAGE-IMPROVEMENT-REFLECTION.md` — Copilot's coverage improvement journal