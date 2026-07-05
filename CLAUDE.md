# Claude Code Context

## Active project: ML LoRA fine-tune
Picking up a Qwen3.6-27B LoRA fine-tune mid-run.

- Repo: https://github.com/nathanaelguitar/ML_Class_LORA (tylerfuentes/ML_Class_LORA is the fork)
- Notebook: https://colab.research.google.com/drive/16FqOifBDZfeKGQpVevna2bbgNqyool-j
- Run name: `qwen36-27b-wrds-500k-unsloth-gb10-rerun-20260616T2330Z`
- Last checkpoint: checkpoint-9500 (~29% of epoch)
- Target: run 3000-step chunks: 12500 → 15500 → 18500 → 21500 → 24500 → 27500 → 30500 → 32750
- Run eval after each chunk, check for regressions before next chunk
- Do NOT push to Hugging Face until explicitly decided

## Google Drive MCP
- Configured as `gdrive` server, authenticated as tfilly93@gmail.com
- If token expires: `GDRIVE_OAUTH_PATH=/Users/tfilly/.claude/gdrive-credentials.json GDRIVE_CREDENTIALS_PATH=/Users/tfilly/.claude/gdrive-token.json npx -y @modelcontextprotocol/server-gdrive auth`
- Notebook file ID: `16FqOifBDZfeKGQpVevna2bbgNqyool-j`

## Key gotchas
- Silent VM reset in Colab: run a new cell — if it executes immediately instead of queuing, training died
- Thinking-mode eval needs --max-new-tokens 512+, use non-thinking mode for reliable eval
- Do NOT rerun fix_checkpoint_key_drift.py on checkpoint-9500 or later
- training/safety.py fingerprint check will hard-fail on bad resume — investigate, don't bypass
