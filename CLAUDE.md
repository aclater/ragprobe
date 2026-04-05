# ragprobe

Adversarial testing and autonomous prompt tuning agent for ragpipe.

## Architecture
```
agent.py loop:
  1. npx promptfoo eval -o results.json
  2. Parse failures by category (66 tests, 13 categories)
  3. Build analysis prompt: current prompt + failures + history
  4. Call local coder LLM (Qwen3-Coder-30B-A3B) for improved prompt
  5. Write prompt + SCP to remote targets
  6. POST /admin/reload-prompt on all targets
  7. Re-run eval → compare
  8. Keep if improved, revert if regressed
  9. Append to history.json for learning
```

## Key files
```
agent.py                    — tuning loop (eval → brain → reload → verify)
ragpipe_provider.py         — custom promptfoo provider returning rag_metadata
assertions/grounding.py     — 7 assertion functions, 40+ multilingual refusal markers
promptfooconfig.js          — dynamic provider list from targets.yaml
prompts/system-prompt.txt   — current best prompt (committed)
targets.yaml                — gitignored, real URLs + tokens + agent config
history.json                — gitignored, iteration history for learning
tests/*.yaml                — 13 test files, 66 adversarial tests
scripts/reload-and-eval.sh  — manual reload + eval
```

## Agent config (in targets.yaml)
```yaml
agent:
  brain_url: http://host:8080       # coder LLM direct (NOT through ragpipe)
  brain_model: qwen3-coder
  max_iterations: 5
  target_pass_rate: 0.85
  prompt_file: prompts/system-prompt.txt
```

## Running
```bash
python agent.py --dry-run              # analyze only
python agent.py --max-iterations 1     # single iteration
python agent.py                        # full 5-iteration run
npx promptfoo eval                     # manual test run
```

## How metadata reaches assertions
Provider returns `{"output": content, "metadata": rag_metadata}`.
Assertions access it via `context["providerResponse"]["metadata"]`.

## Prompt constraints
Brain is instructed to keep prompts under 800 chars. Oversized prompts
(>2000 chars) are truncated in code. The prompt must include citation
format [doc_id:chunk_id] and warning prefix rules.


## GPU acceleration

- This system may have an AMD, NVIDIA, or Intel GPU. All services and scripts must detect the available GPU at runtime and select the appropriate acceleration stack — do not hardcode a vendor.
- Detection priority: NVIDIA CUDA > AMD ROCm > Intel XPU/OpenVINO > CPU. Fall back to CPU only when no GPU is available, and log a clear warning when doing so.
- Never default to CPU for any workload that can run on GPU. CPU fallback is acceptable only when a specific library or operation has no GPU support, and must be explicitly noted in a comment explaining why.
- For Python workloads: use torch.cuda.is_available(), torch.version.hip (ROCm), or torch.xpu.is_available() (Intel) to detect and select the correct device at runtime. Do not hardcode "cuda", "rocm", or "cpu".
- For ONNX Runtime: select ExecutionProvider based on runtime detection — CUDAExecutionProvider, ROCMExecutionProvider, OpenVINOExecutionProvider, or CPUExecutionProvider — in that priority order.
- For container workloads:
  - NVIDIA: pass --device /dev/nvidia0 (or --gpus all with nvidia-container-toolkit)
  - AMD ROCm: pass --device /dev/kfd --device /dev/dri
  - Intel: pass --device /dev/dri
  - Document any container that cannot use GPU and why.
- AMD ROCm on gfx1151: HSA_OVERRIDE_GFX_VERSION=11.5.1 is required. Set this env var in any quadlet, container, or script that uses ROCm on this hardware.
- Do not recommend or implement CPU-only solutions without first investigating whether a GPU-accelerated alternative exists for all three vendors.
- When benchmarking or profiling, always compare GPU vs CPU and report both. Never present CPU-only results as the baseline.
- When writing GPU detection code, always write it once as a shared utility function — do not duplicate vendor detection logic across files.


## Always verify current versions before using them

This is a hard requirement, not a suggestion. Using stale version numbers
wastes time, breaks builds, and has caused real incidents on this stack.

- BEFORE referencing any version number — for a container image, Python
  package, ROCm release, CUDA toolkit, npm package, system package, LLM
  model, or any other software — look it up. Do not use version numbers
  from training knowledge. They are outdated.
- For container images: check the registry (quay.io, ghcr.io,
  registry.access.redhat.com, docker.io) for the current stable tag
  before writing it. Verify the tag exists. Never use :latest in
  production quadlets.
- For Python packages: check PyPI for the current stable release
  before pinning.
- For ROCm: check https://rocm.docs.amd.com and
  https://github.com/RadeonOpenCompute/ROCm/releases for the current
  stable release. ROCm versions change frequently and using an old
  version is a primary cause of GPU acceleration failures on this stack.
- For CUDA: check https://developer.nvidia.com/cuda-downloads for the
  current stable release.
- For npm packages: check https://www.npmjs.com or run
  npm show <package> version.
- For LLM models: check Hugging Face and the model provider directly
  for current releases.
- For system packages (dnf/rpm/apt): do not pin versions unless
  explicitly asked — let the package manager resolve current stable.
- If you cannot verify a version, say so explicitly and ask.
  Do not guess. Do not use what you think the version is.


## Repository location

All code, projects, and repositories live exclusively under ~/git/.

- Never clone, create, or initialize a repository anywhere else on this
  system — not in ~/, not in /tmp, not in ~/Documents, or any other path.
- Before cloning or creating any repo, verify the target path is under
  ~/git/. If it is not, stop and correct the path.
- If you find a repository outside ~/git/, do not work in it. Move it
  to ~/git/ first, update any remotes if needed, and confirm the old
  location is removed before proceeding.
- When referencing local repos, always use ~/git/<reponame> as the path.


## User scripts and tools

User scripts and tools live in ~/.local/bin/, not ~/bin/.

- Always install scripts to ~/.local/bin/
- When referencing or running user scripts, always use ~/.local/bin/<script>
- Never create or reference scripts in ~/bin/ — that path is not used on
  this system


## Working directory conventions

All git repositories and working directories must follow this structure:

- `~/git/` — permanent repositories only. Clone repos here when you intend
  to work in them long-term. Never create temporary work here.
- `~/git-work/<task-name>/` — temporary clones for PR work. Create a
  subdirectory named after the task (e.g. ~/git-work/fix-qdrant-ipv6/).
  Clean up after the PR is merged.
- `~/.local/bin/` — user scripts and tools. Never use ~/bin/.
- Never create git-* directories directly in ~/. They clutter the home
  directory and never get cleaned up.

When starting any task that requires cloning repos:
```bash
mkdir -p ~/git-work/
cd ~/git-work/
gh repo clone aclater/
```

When the PR is merged, clean up:
```bash
rm -rf ~/git-work/
```

Or run the cleanup script periodically:
```bash
~/.local/bin/cleanup-git-work.sh --dry-run
~/.local/bin/cleanup-git-work.sh
```
