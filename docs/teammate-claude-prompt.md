# Prompt for teammate's Claude session

Paste this as your first message to Claude Code in the `ML_Class_LORA` repo directory.

---

You are picking up a Qwen3.6-27B LoRA fine-tune that was handed off mid-run. Here is everything you need to know.

## Repo

`https://github.com/nathanaelguitar/ML_Class_LORA` — already cloned locally. Read `AGENTS.md` and `docs/training-handoff-2026-06-23.md` first for full context. This message is a condensed version of those docs.

## What we are doing

Fine-tuning `Qwen/Qwen3.6-27B` with Unsloth QLoRA (4-bit, LoRA r=16 alpha=32) on ~500k WRDS IBES analyst-revision examples for structured JSON event classification. One epoch = 32,750 steps (effective batch 16).

## Where we are

- **Last checkpoint on Drive: `checkpoint-9500`** (~29% of one epoch done)
- Also present: `checkpoint-9000`
- Run name: `qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z`
- Checkpoint dir: `/content/drive/MyDrive/ML_Class_LORA/checkpoints/qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z/`
- ~23,250 steps and ~8 chunks of 3000 steps remaining

## Eval results so far (non-thinking mode, 50-example WRDS holdout)

| Checkpoint | Accuracy | Macro F1 |
|---|---|---|
| Base Qwen | 0.56 | 0.42 |
| checkpoint-4500 | 1.00 | 1.00 |
| checkpoint-8000 | 1.00 | 1.00 |

Do NOT push any checkpoint to Hugging Face yet — see `docs/colab-a100-storage-policy.md`.

## How to resume

1. Open the notebook on an A100 or H100 runtime (H100 preferred, ~2-3× faster):
   `https://colab.research.google.com/github/nathanaelguitar/ML_Class_LORA/blob/main/notebooks/colab_a100_unsloth_qwen_finance.ipynb`

2. Run all setup cells in order (runtime check → install → kernel restart → mount Drive → clone repo → submodule init → sanity gate). The install cell calls `os._exit(0)` to force a kernel restart — this is intentional, do not skip it.

3. After Drive mounts, verify the highest checkpoint:
   ```python
   import os
   ckpt_dir = "/content/drive/MyDrive/ML_Class_LORA/checkpoints/qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z"
   checkpoints = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")], key=lambda x: int(x.split("-")[1]))
   print("Highest checkpoint:", checkpoints[-1])
   ```

4. Launch a 3000-step chunk. `--max-steps-override` is ABSOLUTE, not a delta. From checkpoint-9500:
   ```python
   RUN_NAME = 'qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z'
   !cd /content/ML_Class_LORA && git pull --ff-only origin main
   !python scripts/colab/train_unsloth_from_config.py \
       --config config/colab_paths.example.yaml \
       --run-name {RUN_NAME} \
       --resume-latest \
       --max-steps-override 12500
   ```
   Next chunk: `--max-steps-override 15500`, then 18500, 21500, 24500, 27500, 30500, 32750.

5. After each chunk completes, run eval and check for regressions before launching the next chunk.

## Critical bugs already fixed in main — do not revert

- `training/common.py:92` — tokenizer call uses `text=` keyword (positional arg silently routed to `images=` in `Qwen3VLProcessor`)
- `training/train_finance_lora_unsloth.py:149` — unwraps `Qwen3VLProcessor` to `.tokenizer` before building dataset/collator (wrapper has no `.pad()`)
- `training/fix_checkpoint_key_drift.py` — one-time LoRA key repair already applied to checkpoint-4500; **do NOT rerun on checkpoint-9500 or later, it would corrupt the weights**
- `scripts/colab/train_unsloth_from_config.py:152` — `--max-steps-override` passthrough

## Gotchas

- **Silent VM reset:** Colab can die with no visible error. The training cell will look like it is still running (stale null execution_count). Test by adding a new cell and running it — if it executes immediately instead of queuing, the kernel is free and training has died. Drive checkpoints are always safe.
- **FinGPT submodule:** `/content` is ephemeral. After every VM reset, `git submodule update --init --recursive` must run before training. The notebook setup cell does this — do not skip cells.
- **Backup dir:** There is a pre-fix backup of checkpoint-4500 at `...checkpoints/qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z-checkpoint-4500-backup-keydrift/` — this is intentionally outside the run dir. Do not move it inside the run dir or `latest_checkpoint()` will crash trying to parse "keydrift" as an integer.
- **Thinking mode eval:** Always use `--max-new-tokens 512+` for thinking mode. Default 96 tokens always truncates thinking output before the JSON answer, producing `parse_failure_rate: 1.0` that looks like a regression but is just a budget issue. Non-thinking mode is the reliable eval path.
- **`training/safety.py` fingerprint check:** Will hard-fail on resume if LoRA keys don't match the checkpoint. This is intentional protection. If it fires, do not bypass it — investigate the key mismatch.

## Key docs to read

- `AGENTS.md` — full project context, directory map, hard rules
- `docs/training-handoff-2026-06-23.md` — this handoff in full
- `docs/eval-findings.md` — complete eval history and model selection rationale
- `docs/colab-a100-storage-policy.md` — what goes to HF and when
- `docs/google-colab-browser-quickstart.md` — Colab setup walkthrough
