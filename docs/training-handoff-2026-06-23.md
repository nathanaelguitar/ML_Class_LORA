# Training Handoff — 2026-06-23

**Written for:** teammates picking up the WRDS-500k Unsloth fine-tune run  
**Handed off by:** Nathanael  
**Status:** training is paused, compute is exhausted; resume when you have A100/H100 credits

---

## What we are training

| Field | Value |
|---|---|
| Model | `Qwen/Qwen3.6-27B` |
| Method | Unsloth QLoRA (4-bit), LoRA r=16 alpha=32 |
| Task | WRDS IBES analyst-revision event classification (structured JSON output) |
| Run name | `qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z` |
| Dataset | ~500k WRDS IBES examples on Google Drive (`data/gold/train.jsonl`) |
| Epochs | 1 |
| Effective batch | 16 (per_device=4 × grad_accum=4) |
| **Total steps** | **32,750** |
| Learning rate | 2e-4 |
| Save interval | every 500 steps |

---

## How far we got

**Last confirmed checkpoint on Drive: `checkpoint-9500`** (saved 2026-06-23 ~6:53 AM)

Also present: `checkpoint-9000` (saved ~3:55 AM). A chunk targeting step 11000 was launched but the runtime died at step 9500.

```bash
ls /content/drive/MyDrive/ML_Class_LORA/checkpoints/qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z/
```

The highest `checkpoint-NNNNN` directory is where you resume from.

**Progress:** 9500 / 32750 steps = **~29% of one epoch**

---

## Evaluation results so far

All evals use 50-example WRDS holdout, non-thinking mode, `--max-new-tokens 96`.

| Checkpoint | Accuracy | Macro F1 | Parse fail rate |
|---|---|---|---|
| Base Qwen (no adapter) | 0.56 | 0.42 | 0.04 |
| checkpoint-4500 (71k examples) | 1.00 | 1.00 | 0.00 |
| checkpoint-8000 | 1.00 | 1.00 | 0.00 |

The adapter is strong — perfect on this holdout from step 4500 onward. We are continuing training to improve generalization beyond this 50-example window and toward a larger eval that would justify HF promotion.

**Do not push any checkpoint to Hugging Face yet.** See `docs/colab-a100-storage-policy.md`.

---

## How to resume

### 1. Open the notebook on an A100 or H100 runtime

```
https://colab.research.google.com/github/nathanaelguitar/ML_Class_LORA/blob/main/notebooks/colab_a100_unsloth_qwen_finance.ipynb
```

H100 is preferred — roughly 2-3× faster than A100 for this workload.

### 2. Run cells in order through the setup section

The notebook cells handle: runtime check → install deps → kernel restart → mount Drive → clone repo → submodule init → sanity gate.

> **Gotcha:** After the install cell, the kernel **must be restarted** (cell calls `os._exit(0)`) before continuing. This is intentional — Unsloth downgrades torch and the stale import must be flushed.

### 3. Check the highest checkpoint

After Drive mounts, run:

```python
import os
ckpt_dir = "/content/drive/MyDrive/ML_Class_LORA/checkpoints/qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z"
checkpoints = sorted([d for d in os.listdir(ckpt_dir) if d.startswith("checkpoint-")],
                     key=lambda x: int(x.split("-")[1]))
print("Highest checkpoint:", checkpoints[-1] if checkpoints else "none")
```

### 4. Launch the next training chunk

Use **3000-step chunks** to stay well under Colab's rate-limit / disconnect window. The `--max-steps-override` flag is **absolute** (not a delta).

If the highest checkpoint is `checkpoint-N`, pass `--max-steps-override N+3000`:

```python
RUN_NAME = 'qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z'

# Resuming from checkpoint-9500 → target step 12500
!cd /content/ML_Class_LORA && git pull --ff-only origin main
!python scripts/colab/train_unsloth_from_config.py \
    --config config/colab_paths.example.yaml \
    --run-name {RUN_NAME} \
    --resume-latest \
    --max-steps-override 12500
```

Repeat with N+3000 each successful chunk until you reach step 32750.

### 5. After each chunk: run eval before the next one

```python
!python scripts/colab/run_eval_from_config.py \
    --config config/colab_paths.example.yaml \
    --run-name {RUN_NAME} \
    --resume-latest
```

Check that accuracy/F1 have not regressed before proceeding.

---

## Code bugs already fixed (all in main)

These were real blockers in earlier sessions. **Do not revert or work around them.**

| File | Fix |
|---|---|
| `training/common.py:92` | Pass `text=` as keyword arg to tokenizer — positional arg routed to `images=` slot of `Qwen3VLProcessor` |
| `training/train_finance_lora_unsloth.py:149` | Unwrap `Qwen3VLProcessor` to `.tokenizer` before building dataset/collator — the wrapper lacks `.pad()` |
| `training/fix_checkpoint_key_drift.py` | One-time repair script for the `language_model.`/`.default` key drift in `checkpoint-4500`; **already applied, do not rerun on checkpoint-8000+** |
| `scripts/colab/train_unsloth_from_config.py:152` | `--max-steps-override` passthrough to the training script |

### LoRA key drift — already fixed, do not re-fix

`checkpoint-4500` had a structural key mismatch between how Unsloth saved the adapter and how PEFT expected to load it. This was repaired in-place with `training/fix_checkpoint_key_drift.py` and all subsequent checkpoints (`checkpoint-8000`, etc.) were saved correctly by the training loop.

**Do not run `fix_checkpoint_key_drift.py` on checkpoint-8000 or later.** It would corrupt correct weights.

The `training/safety.py` fingerprint check (`verify_resume_fingerprint`) will hard-fail if keys are mismatched — this is intentional protection, not a bug.

---

## Drive directory layout

```
/content/drive/MyDrive/ML_Class_LORA/
├── checkpoints/
│   └── qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z/
│       ├── checkpoint-4500/          # repaired (key drift fix applied)
│       ├── checkpoint-8000/          # last confirmed good checkpoint
│       └── checkpoint-11000/         # may or may not exist — check first
├── data/gold/
│   ├── train.jsonl                   # ~500k examples
│   ├── train_eval.jsonl
│   └── test.jsonl
├── adapters/                         # final adapter copies (post-run)
├── outputs/train_logs/               # per-run .log files
└── manifests/                        # per-run JSON manifests
```

---

## What to watch out for

**Silent VM reset:** Colab can kill the VM with no visible error. The training cell will appear to still be running (stale `execution_count: null`) while the kernel is actually free. To test: add a new cell and run it. If it executes immediately instead of queuing, the kernel is free and training has died. Drive checkpoints from before the reset are safe.

**Backup dir naming:** There is a backup of the original (pre-fix) `checkpoint-4500` at:
```
.../checkpoints/qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z-checkpoint-4500-backup-keydrift/
```
This is **outside** the run directory on purpose — the `latest_checkpoint()` function in the wrapper uses `int(item.name.split("-")[-1])` and would crash on a directory named `checkpoint-4500-backup-keydrift`.

**FinGPT submodule:** `/content` is ephemeral. After each VM reset, re-run:
```bash
cd /content/ML_Class_LORA && git submodule update --init --recursive
```
The notebook setup cell handles this, but if you skip cells, training will fail with a `FinGPT submodule not found` error.

**Thinking mode:** Do not evaluate in thinking mode without `--max-new-tokens 512+`. Thinking-mode outputs are verbose and always truncate at the default 96-token budget, producing `parse_failure_rate: 1.0` that looks like a regression but is just truncation. Non-thinking mode is the reliable eval path.

---

## Remaining work

- Steps remaining: **~23,250** (32750 − 9500)
- Chunks of 3000 remaining: **~8 more**
- After training completes: run a larger eval (≥200 examples) before any HF promotion decision
- See `docs/eval-findings.md` for the full eval history and model-selection rationale
- See `docs/colab-a100-storage-policy.md` for the promotion-to-HF policy
