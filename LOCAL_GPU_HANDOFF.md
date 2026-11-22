# FlexGlue: local GPU handoff

Prepared 21 November 2026. Start with this page, the adjacent `tools/local_gpu/`
directory, and the current PR bodies. No earlier artifact archive is required.

[Current local GPU queue](https://github.com/fringehush-gif/FlexGlue/pulls?q=is%3Apr+is%3Aopen+label%3Aneeds-local-gpu)

## What changed while this workstation was offline

- Current review owners replace closed aggregate PRs **#10/#13/#14/#15/#16/#28**.
  Follow the stacks below; do not replay those older bundles.
- **#50 is closed into #12 / existing upstream #26.** Its singleton-resize behavior
  already existed there. The stronger regression now lives in the existing test
  file; 13 maintained methods passed. The older measurements remain evidence for
  that optimization, not an additional gain over #12.
- **#27 → #31 → #32 → #33** now has current parent ancestry throughout. The final
  maintenance changes tests only: copied descriptors retain their default
  precision setting, CUDA comparisons require complete outputs, and the RGB8
  fixture supplies generated weights through the real checkpoint reader.
- **#17 → #44** now explicitly stacks the distance-bound fix and dense-cycle
  optimization. Its regression exercises real geometry rather than a helper mock.
- Existing upstream **#29** and fork **#27** have identical complete trees. The
  unused export helper is removed; the required `mp=None` fixture setting remains.
- **#42** now supplies that same default in its image-size fixture. The combined
  checkpoint/RGB8/image-size diagnostic suite passes 105 CPU methods, with 22
  device or optional methods skipped. This is no new GPU or performance result.
- Fork `main` remains a separate, older integration line. **#21/#34/#35/#37** retain
  main-target CI/backport work. **#45** remains a small independent entmax
  correctness proposal overlapping the readout family. Do not apply it twice.
- **Twelve obsolete branches were removed** after checking all active fork and
  upstream heads and bases. Eleven closed PR histories retain the exact former
  commits; the old base is already in `main` ancestry. The seven GPU profiles
  below identify the maintained review heads.
- **No new upstream PRs.** Keep existing review units small; update an existing
  upstream counterpart where appropriate. Use the current review owners above.

## First: use the existing working environment

Start in the previously working WSL CUDA/Triton environment on the RTX 3080 Ti.
Use the WSL Linux filesystem for checkouts and results. Record the actual GPU,
driver, Python, Torch, CUDA runtime and Triton versions; do not infer them from
old receipts. No driver or package upgrade is required by this handoff. Resolve
missing dependencies deliberately without replacing a working Torch build.

~~~bash
nvidia-smi
python -c 'import sys, torch, triton; print(sys.version); print("Torch", torch.__version__, "CUDA", torch.version.cuda, "Triton", triton.__version__); print("CUDA available", torch.cuda.is_available()); print(torch.cuda.get_device_name(0))'
~~~

Use an idle, explicitly selected GPU. Record `CUDA_VISIBLE_DEVICES` if set.
Do not run with `python -O`; qualification requires assertions.

## Make clean, separate target worktrees

The maintenance checkout supplies the runner; each target PR supplies its own
production code and maintained tests. Use one process per target checkout.
Preserve existing local work. Inspect it first; do not reset, clean or overwrite it.
Choose a new empty run directory and replace the two example paths below.

~~~bash
set -e
flexglue_repo=/path/to/your/FlexGlue
flexglue_runs=/path/to/new/flexglue-local-20261121
git -C "$flexglue_repo" status --short
git -C "$flexglue_repo" worktree list
mkdir -p "$flexglue_runs/results"
git -C "$flexglue_repo" fetch --no-tags https://github.com/fringehush-gif/FlexGlue.git refs/heads/codex/local-gpu-handoff-20261121
git -C "$flexglue_repo" worktree add --detach "$flexglue_runs/tooling" FETCH_HEAD
flexglue_runner="$flexglue_runs/tooling/tools/local_gpu/qualify.py"
for flexglue_pr in 19 22 23 24 25 26 36; do
  git -C "$flexglue_repo" fetch --no-tags https://github.com/fringehush-gif/FlexGlue.git "pull/$flexglue_pr/head"
  git -C "$flexglue_repo" worktree add --detach "$flexglue_runs/pr$flexglue_pr" FETCH_HEAD
done
nvidia-smi > "$flexglue_runs/results/nvidia-smi.txt"
~~~

The explicit fork URL prevents fetching an unrelated upstream PR with the same
number. `PROFILES` in `qualify.py` pins full reviewed commits. A changed live head
must be reviewed before using `--expected-head`; do not bypass a mismatch merely
to make the run pass. Keep executable edits in reviewed commits, not dirty trees.

## Four campaigns, seven review heads

| Order | PR / pinned head prefix | Required question |
| --- | --- | --- |
| 1 | #19 `6d7b24d23b05` | CUDA KDE with native tent and blur arithmetic, delayed addresses, and seeded sampler input/RNG ownership. Deterministic correctness is separate from default-scatter variability. |
| 2 | #22 `91d0aea051a8` | Native, unpromoted selected-root differentiation and the recovered BF16 actual-head fixture. |
| 2 | #23 `a62db4d54f5e` | Separate FP16/BF16-to-FP32 policy, confidence behavior and the original FP16 CUDA failure regression. |
| 2 | #24 `d801d271920f` | Readout stack plus actual eligible Triton launch, rectangular exact-512 fixture and fallback/gradient/compiled delegation. |
| 3 | #25 `0aae87fdd1ea` | Native refiner reduction layout through actual consumers and gradients, including mixed precision and compiled feature-cache lifetime. |
| 3 | #26 `15aa1c6007f9` | Native/fused/fallback refinement composition and actual Triton backend regressions. |
| 4 | #36 `1d98a6fb9f12` | Stored FP32 precision inversion in maintained tests, actual two-stage consumer and fullgraph-compiled forward. |

Run the profiles in that order, stopping to investigate a failure:

~~~bash
set -e
for flexglue_pr in 19 22 23 24 25 26 36; do
  python "$flexglue_runner" --pr "$flexglue_pr" \
    --repo "$flexglue_runs/pr$flexglue_pr" \
    --output "$flexglue_runs/results/pr$flexglue_pr.json" \
    > "$flexglue_runs/results/pr$flexglue_pr.log" 2>&1
done
~~~

CUDA is the default. Required backend tests must execute without skips. Missing
CUDA/Triton, wrong source or dirty executable code does not count as qualification.
Compiled-consumer checks require active Torch compilation, with Triton's CPU
interpreter disabled for CUDA/Triton profiles.
`--device cpu` is a harness smoke check only and cannot clear a GPU label. Read
the JSON and log, not only the process exit code. A CPU smoke can exit successfully.

Share common checks within a family only after establishing source equivalence.
Keep results for #22, #23 and #24 separately attributable: #24 inherits FP32
promotion and cannot qualify #22's unpromoted BF16 arithmetic. Current complete
stacks are **#17 → #44**, **#22 → #23 → #24**, **#25 → #26**, **#29 → #30**, and
**#27 → #31 → #32 → #33**. These are review families, not instructions to combine
all tips into a default runtime.

## Inputs, measurements and remaining holds

The required runner uses generated inputs and model state. It needs **no external
scene, downloaded checkpoint or pretrained model**. The bundled 1.84 MB BF16
fixture preserves a recovered failure case independently of device RNG; its
digest is checked and it is loaded with `weights_only=True`.

Real-consumer timing and quality work is separate. Previous work used
**Fountain-P11 at 1536 and 3072 px**, with `flexglue.pth`, `romav2.0.1.pt` and real
image pairs. Those assets are optional for the generated gates. Follow the
current checkout's README/loader for checkpoint provenance; an old experiment
archive is unnecessary. Generated gates do not establish reconstruction quality.

After correctness, measure only the claims under consideration: FP32/FP16/BF16,
same inputs/settings, synchronized warmed GPU runs, an A/A noise control and
interleaved baseline/candidate runs. Distinguish allocated peak, reserved memory,
process RSS and allocation volume. Do not attach historical figures to a new
combined tree. Optional RGB8/Triton paths have mixed or adverse timing evidence;
keep them optional and do not flip defaults based on a microbenchmark.

**#36** passed the generated two-stage and compiled-forward CUDA checks. The
separate learned-pair replay checks inversion of the stored precision at two
corrected cells. See the PR for its measured timing and scope. **#47** tracks
the independent CPU lifetime and latency investigation.

## Return results and advance the PRs

Attach concise results to the relevant existing PR: tested head and clean-source
identity, hardware/software, command, executed/skipped counts, numerical outcome,
and measured claims with their controls. Fix failures on the owning review unit
and propagate shared fixes through its actual dependents.

Remove `needs-local-gpu` only after the PR's stated required checks have evidence
for its current production code, or demonstrated source-equivalent code. A runner
pass covers its bounded fixtures, not every API/quality/performance decision.
Keep downstream and upstream descriptions consistent.

