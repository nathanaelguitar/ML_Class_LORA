# Qwen3.6-27B QLoRA Fine-Tune — Progress Report

**Run name:** `qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z`  
**Last updated:** 2026-07-05  
**Current checkpoint:** 21,500 (~29% of epoch)

---

## Model

| Property | Value |
|---|---|
| Base model | Qwen/Qwen3.6-27B (Alibaba) |
| Quantisation | 4-bit QLoRA (NF4 + double quantisation) |
| LoRA rank | 64, alpha 128 |
| Target modules | q/k/v/o projections |
| Context length | 8,192 tokens |
| Framework | Unsloth + PEFT 0.19.1 + bitsandbytes 0.49.2 |
| Compute precision | bf16 compute, fp16 output |

---

## Training Configuration

| Property | Value |
|---|---|
| Hardware | Google Colab A100-SXM4-80 GB High-RAM |
| System RAM | 83 GB |
| Dataset | WRDS multi-source financial corpus, 500K examples |
| Batch size | 2 per device × 4 gradient accumulation = 8 effective |
| Learning rate | 2e-4, cosine schedule, 500-step warm-up |
| Chunk size | 3,000 steps per training segment |
| Target total steps | 32,750 |

---

## Checkpoint Timeline

| Checkpoint | Status | Notes |
|---|---|---|
| 0 | ✅ Done | Training start |
| 9,500 | ✅ Done | Chunk 1 complete; key drift fix applied (do not re-run fix script) |
| 12,500 | ✅ Done | Chunk 2 complete |
| 15,500 | ✅ Done | Chunk 3 complete |
| 18,500 | ✅ Done | Chunk 4 start |
| **21,500** | **🔄 Eval running** | **Current — chunk 4 eval in progress** |
| 24,500 | ⏸ Queued | Chunk 5 — pending GO/NO-GO |
| 27,500 | ⏸ Queued | Chunk 6 |
| 30,500 | ⏸ Queued | Chunk 7 |
| 32,750 | ⏸ Queued | End of epoch |

---

## Infrastructure Built

### Storage & Sync
- Checkpoints written to Google Drive at `/ML_Class_LORA/checkpoints/<run-name>/`
- Eval results written to `/ML_Class_LORA/eval_results/chunk<N>_<step>/`
- Drive mounted as `/content/drive` inside Colab VM; survives notebook saves

### Codebase
- Repo: `nathanaelguitar/ML_Class_LORA` (fork: `tylerfuentes/ML_Class_LORA`)
- FinGPT submodule at `external/FinGPT` (21 items, must re-init after every VM reset)
- `training/safety.py` — fingerprint check hard-fails on bad checkpoint resume (intentional)
- `eval/evaluate_base_vs_adapter.py` — runs base + adapter side-by-side, 200 examples

### Eval Pipeline (`run_evals.sh`)
Runs 6 benchmarks sequentially against a given checkpoint:
1. WRDS holdout (`data/gold/test.jsonl`) — in-domain accuracy
2. FPB — Financial PhraseBank sentiment
3. FIQA — Financial opinion QA
4. TFNS — Twitter financial news sentiment
5. NWGI — News with genuine intent
6. Headline — Gold commodity headline classification

All benchmarks: non-thinking mode, `--max-new-tokens 80`, greedy decoding, 200 examples each.

### Dependency Stack (required on every fresh VM)
1. Mount Drive — `drive.mount('/content/drive', force_remount=True)`
2. Clone repo — `git clone https://github.com/nathanaelguitar/ML_Class_LORA`
3. Init FinGPT — `git submodule update --init --recursive external/FinGPT`
4. Uninstall torchao — incompatible with torch 2.6.0
5. Patch `peft/utils/constants.py` — `BloomPreTrainedModel` removed in transformers 5.5.0
6. Install bitsandbytes — `pip install -U bitsandbytes>=0.46.1` (not pre-installed)

---

## Chunk 4 Eval — In Progress (2026-07-05)

| Benchmark | Status | Log |
|---|---|---|
| WRDS holdout | 🔄 Running | `eval_results/chunk4_21500/wrds_holdout.log` |
| FPB | ⏸ Queued | — |
| FIQA | ⏸ Queued | — |
| TFNS | ⏸ Queued | — |
| NWGI | ⏸ Queued | — |
| Headline | ⏸ Queued | — |

- Run started: 2026-07-05 17:15 UTC
- Background PID: 29548 (session leader)
- Model loaded: ~17:34 UTC (851/851 weights, 34 GB GPU RSS)
- ETA all benchmarks complete: ~22:00–23:00 UTC

---

## Post-Chunk Decision Protocol

**GO criteria:** All 6 benchmarks at or above chunk 3 baseline.

**REGRESSION mitigations (in order):**
1. Lower LR: 2e-4 → 1e-4
2. Reduce LoRA rank: 64 → 32
3. Mix in general-domain examples
4. Shorten chunk: 3,000 → 1,500 steps

**Hard constraints:**
- Do NOT push to Hugging Face until explicitly decided
- Do NOT rerun `fix_checkpoint_key_drift.py` on checkpoint-9500 or later
- `--max-steps-override` is absolute, not a delta

---

## Presentation Assets

- `docs/Qwen3.6-27B_LoRA_Training_Progress.pptx` — 6-slide deck (also on Desktop)
- `docs/slide_script.txt` — speaker notes for all 6 slides
