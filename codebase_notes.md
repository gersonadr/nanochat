# Nanochat Codebase Notes

> Auto-generated analysis for AI context. Covers the full pipeline for building a chat LLM from scratch on a single GPU node for under $100.

---

## Project Overview

**Nanochat** is a minimal, end-to-end LLM training framework designed to train GPT-2 capability models on a single GPU node for under $100. It covers the complete pipeline: tokenization, pretraining, finetuning, evaluation, inference, and an interactive chat UI.

---

## Directory Structure

```
/home/user/nanochat/
├── nanochat/              # Core library modules
├── scripts/               # Training and inference entry points
├── tasks/                 # Evaluation task implementations
├── dev/                   # Development utilities and data handling
├── tests/                 # Test files
├── runs/                  # Shell scripts for common training recipes
└── README.md              # Project documentation
```

---

## Core Modules (`nanochat/`)

### `gpt.py` — GPT Model Architecture

- **Main class**: `GPT(nn.Module)` — transformer-based language model
- **Key components**:
  - `GPTConfig`: dataclass defining model architecture parameters
  - `CausalSelfAttention`: multi-head attention with Group-Query Attention (GQA) support
  - `MLP`: feed-forward network with ReLU² activation
  - `Block`: transformer block combining attention + MLP
  - `Linear`: custom linear layer that casts weights to COMPUTE_DTYPE on forward pass
- **Notable features**:
  - Rotary embeddings (RoPE) for positional encoding
  - QK normalization in attention
  - Sliding window attention patterns (SSSL pattern)
  - Value embeddings (ResFormer-style) on alternating layers
  - Per-layer learnable scalars (`resid_lambdas`, `x0_lambdas`)
  - Smear gate for mixing previous token embeddings
  - Backout mechanism to remove low-level features
  - Flash Attention 3 integration with SDPA fallback
  - Inference support with KV caching
- **Key methods**:
  - `forward(idx, targets, kv_cache, loss_reduction)`: training and inference
  - `generate(tokens, max_tokens, temperature, top_k)`: autoregressive generation
  - `setup_optimizer()`: builds mixed MuonAdamW optimizer with different LRs per parameter group
  - `estimate_flops()`: estimates FLOPs per token including attention costs

### `tokenizer.py` — BPE Tokenization

- **Two implementations**:
  1. `HuggingFaceTokenizer`: full-featured, trains with HF tokenizers library
  2. `RustBPETokenizer`: trains with rustbpe, inference with tiktoken
- **Special tokens**: BOS, user_start/end, assistant_start/end, python_start/end, output_start/end
- **Key methods**:
  - `encode()`: convert text to token IDs
  - `decode()`: convert token IDs back to text
  - `render_conversation()`: tokenize chat conversations with masks for training
  - `render_for_completion()`: prime for RL generation

### `common.py` — Utilities

- `COMPUTE_DTYPE`: auto-detects precision (bfloat16 for A100/H100, float32 for older GPUs)
- DDP setup: `compute_init()`, `compute_cleanup()`, distributed communication
- Logging: colored logging setup
- File management: `download_file_with_lock()` for downloading datasets
- GPU detection: `get_peak_flops()` for GPU-specific peak performance

### `dataloader.py` — Data Loading

- **Main function**: `tokenizing_distributed_data_loader_with_state_bos_bestfit()`
- **Algorithm**: best-fit packing with BOS alignment
  - Every row starts with BOS token
  - Documents packed to maximize space utilization
  - ~35% token cropping (unavoidable with fixed-length rows)
  - 100% VRAM utilization (no padding)
- **Features**:
  - Distributed Data Parallel (DDP) sharding across ranks
  - Resumable state tracking (parquet index, row group, epoch)
  - Buffered document loading for best-fit selection

### `dataset.py` — Pretraining Data

- **Current dataset**: ClimbMix-400B (NVIDIA dataset)
- **Legacy support**: FinewebEdu-100B
- On-demand downloading from HuggingFace Hub
- Parallel downloads with retry logic, parquet format, DDP sharding

### `engine.py` — Efficient Inference

- **Main class**: `Engine(model, tokenizer)` — handles efficient generation with KV cache
- **KVCache class**:
  - Manages key/value cache for FA3 format (B, T, H, D)
  - Batch generation with position tracking
  - Cache prefilling for multi-sample generation
- **Key methods**:
  - `generate()`: streaming generation with KV cache
  - `generate_batch()`: batch generation returning token sequences
  - `sample_next_token()`: temperature and top-k sampling
- **Tool use**:
  - Python execution via `<|python_start|>` / `<|python_end|>` tokens
  - Safe calculator evaluation with timeout and security checks
  - Output wrapped in `<|output_start|>` / `<|output_end|>` tokens

### `checkpoint_manager.py` — Model Checkpointing

- **Save format**: separate model params, optimizer state (per-rank), and metadata (JSON)
- **Key functions**:
  - `save_checkpoint()`: save model, optimizer, metadata
  - `load_checkpoint()`: load from disk
  - `build_model()`: rebuild model from checkpoint with state
  - `load_model()`: convenience wrapper for base/sft/rl models
- **Backward compatibility**: automatic patching for old checkpoints missing new config keys

### `optim.py` — Optimizer

- **MuonAdamW**: combined optimizer for matrix parameters and embeddings
- **Two optimizers**:
  1. **AdamW** (fused kernel): for embeddings, LM head, scalars
  2. **Muon**: for 2D transformer matrices (polar express orthogonalization)
- **Muon details**: momentum + orthogonalization via Polar Express, variance reduction (NorMuon), cautious weight decay, Newton-Schulz iteration alternative
- **DDP support**: `DistMuonAdamW` for distributed training

### `loss_eval.py` — Evaluation Metrics

- **BPB (Bits Per Byte)**: vocab-size-invariant loss metric
- `evaluate_bpb()`: normalizes loss by token byte length, excludes special tokens and masked targets, distributed reduction across ranks

### `flash_attention.py` — Attention Backends

- Unified interface matching FA3 API
- Auto-detection: uses Flash Attention 3 on Hopper (H100/H200) with bfloat16, falls back to PyTorch SDPA elsewhere
- `flash_attn_func()`: training (no KV cache)
- `flash_attn_with_kvcache()`: inference with cache management
- Sliding window attention + GQA support

### `core_eval.py` — DCLM CORE Metric

- Evaluates language model quality on multiple choice, schema, and LM tasks
- Prompt rendering with few-shot examples
- Compatible with DCLM paper benchmark

### `fp8.py` — FP8 Training

- `Float8Linear`: drop-in `nn.Linear` replacement
- Tensorwise dynamic scaling (one scale per tensor)
- Three FP8 matmuls per forward/backward: forward, backward grad_output@weight, backward grad_output.T@input
- Dtypes: float8_e4m3fn (input/weight), float8_e5m2 (gradients)
- ~2x faster matmuls via cuBLAS

### `execution.py` — Sandboxed Code Execution

- Runs Python code from LLM in isolated process
- Timeout protection, memory limits (256 MB), stdout/stderr capture
- Dangerous functions disabled (os.system, subprocess, etc.)
- Used for code generation evaluation

### `report.py` — Training Reports

- Generates metadata about training runs
- Git info, GPU info, system info collection, used for model card generation

---

## Training Scripts (`scripts/`)

### `base_train.py` — Pretraining

Entry point: `python -m scripts.base_train` or `torchrun --nproc_per_node=8 -m scripts.base_train`

Key hyperparameters:
- `--depth`: number of transformer layers (main complexity dial)
- `--aspect-ratio`: model width = depth × aspect-ratio
- `--max-seq-len`: context window (default 2048)
- `--device-batch-size`: per-GPU batch size
- `--total-batch-size`: global batch size in tokens
- `--window-pattern`: sliding window attention pattern

Training flow:
1. Build model on meta device (shapes only)
2. Move to GPU, initialize weights
3. Create optimizers (MuonAdamW) with different LRs
4. Load data via distributed dataloader
5. Train loop: forward → loss → backward → optimizer step, with gradient accumulation, periodic BPB/CORE evaluation, sampling, checkpointing, LR warmup/warmdown

Features: FP8 training (`--fp8`), resumable checkpoints (`--resume-from-step`), DDP, wandb logging

### `chat_sft.py` — Supervised Fine-Tuning

- Loads pretrained base model, inherits hyperparameters from pretraining
- Task-based training: MMLU, GSM8K, SmolTalk, CustomJSON, SpellingBee
- Can override learning rates and batch sizes
- Multi-epoch training, evaluation on multiple benchmarks

### `chat_rl.py` — Reinforcement Learning

Fine-tunes via reward signals; similar structure to chat_sft.

### `base_eval.py` / `chat_eval.py` — Evaluation

- `base_eval.py`: evaluates BPB and CORE metric on validation set
- `chat_eval.py`: evaluates SFT/RL models on tasks (MMLU, GSM8K, etc.)

### `tok_train.py` / `tok_eval.py` — Tokenizer

- `tok_train.py`: trains BPE tokenizer on pretraining data
- `tok_eval.py`: evaluates tokenizer compression ratio

### `chat_cli.py` — Interactive Chat

- Interactive conversation mode and single prompt evaluation mode (`-p`)
- Temperature and top-k control, conversation history tracking
- Engine-based efficient generation

### `chat_web.py` — Web UI

- ChatGPT-like web interface (embedded HTML/CSS/JS)
- Real-time generation streaming, works with sft or rl models

---

## Evaluation Tasks (`tasks/`)

| File | Dataset | Type |
|------|---------|------|
| `mmlu.py` | MMLU (57 subjects) | Multiple choice |
| `gsm8k.py` | GSM8K (8K problems) | Math + tool use |
| `arc.py` | ARC Challenge/Easy | Science QA |
| `humaneval.py` | HumanEval | Code generation |
| `smoltalk.py` | SmolTalk | Conversational SFT |
| `customjson.py` | Arbitrary JSONL | Custom data |
| `spellingbee.py` | Synthetic | Letter counting |
| `common.py` | — | `Task`, `TaskMixture`, `TaskSequence` base classes |

---

## Key Data Flow & Pipeline

```
Pretraining Pipeline:
├─ Dataset (ClimbMix-400B parquet files)
│  └─ Download on-demand from HuggingFace
├─ Tokenizer (rustbpe + tiktoken)
│  └─ Train on pretraining data, store vocab
├─ Data Loader (BOS-aligned best-fit packing)
│  └─ DDP-sharded, resumable
├─ Model (GPT with sliding windows, value embeddings, etc.)
│  └─ Initialize on meta device, move to GPU
├─ Optimizer (MuonAdamW with mixed precision)
│  └─ AdamW for embeddings, Muon for matrices
├─ Training Loop
│  ├─ Forward pass → loss computation
│  ├─ Backward pass → gradients
│  ├─ Optimizer step with gradient accumulation
│  ├─ Periodic evaluation (BPB, CORE metric)
│  └─ Checkpointing
└─ Output: Base checkpoint (model params + metadata)

Supervised Fine-Tuning Pipeline:
├─ Load base checkpoint
├─ Load conversation datasets (MMLU, GSM8K, etc.)
├─ Mix tasks deterministically
├─ Tokenize with conversation rendering
│  └─ Special tokens mark user/assistant boundaries
├─ Train with task mixture
│  └─ Only assistant tokens contribute to loss
└─ Output: SFT checkpoint

Inference Pipeline:
├─ Load checkpoint (base/sft/rl)
├─ Initialize Engine with model + tokenizer
├─ User provides prompt tokens
├─ KV cache prefill on prompt
├─ Autoregressive sampling with:
│  ├─ Temperature control
│  ├─ Top-k filtering
│  ├─ Tool use (python execution)
│  └─ KV cache for efficiency
└─ Stream tokens to user
```

---

## Model Architecture Details

**Depth-driven complexity**: single `--depth` parameter controls number of layers, model width, number of heads, all learning rates (scaled by √dim), and training duration (data:param ratio maintained).

**Current configuration (d24–d26 range)**:
- ~1.3–1.6B parameters
- GPT-2 capability level
- Trainable in 2–3 hours on 8×H100
- CORE score ~0.258–0.269

**Attention patterns**: alternating sliding window + full context ("SSSL" pattern), reduces computation for early layers, last layer always full context, QK normalization for stability.

---

## Hyperparameter Groups

| Parameter Group | Optimizer | Learning Rate | Use Case |
|---|---|---|---|
| Transformer matrices (attention, MLP) | Muon | 0.02 | Main network training |
| Token embedding (wte) | AdamW | 0.3 | Input representation |
| LM head (unembedding) | AdamW | 0.004–0.008 | Output logits |
| Per-layer scalars (resid_lambdas) | AdamW | 0.005 | Residual scaling |
| x0_lambdas | AdamW | 0.5 | Input blending |
| Smear/backout gates | AdamW | 0.2 | Feature mixing |

---

## Notable Techniques

1. **Value Embeddings (ResFormer)**: alternating layers have learnable value embeddings per vocab token
2. **Smear Gate**: mix previous token's embedding into current position (cheap bigram info)
3. **Backout**: subtract cached mid-layer residual before final norm to remove low-level features
4. **Per-layer Scalars**: learnable per-layer multipliers for residual stream and input blending
5. **QK Normalization**: stabilize attention logits
6. **FP8 Training**: optional ~2× speedup via tensorwise quantization
7. **Sliding Window Attention**: reduce computation in early layers
8. **Flash Attention 3**: hardware-specific optimization (Hopper) with SDPA fallback
9. **Best-fit Packing**: minimize token waste while maintaining 100% VRAM utilization

---

## Evaluation Metrics

- **BPB (Bits Per Byte)**: vocab-size-invariant loss (normalized by token bytes)
- **CORE Score**: DCLM benchmark covering multiple NLP tasks
- **Task-specific metrics**: accuracy for multiple choice, execution pass rate for code
- **MFU (Model FLOPs Utilization)**: training efficiency metric

---

## Distributed Training

- **DDP (Distributed Data Parallel)**: multi-GPU training via torch.distributed
- **Rank-aware data sharding**: each rank gets different row groups
- **Resumable checkpoints**: per-rank optimizer state, shared model state
- **Gradient accumulation**: effective large batch sizes on limited VRAM
- **All-reduce**: distributed evaluation metrics

---

## Utilities & Run Scripts (`dev/`, `runs/`)

| Script | Purpose |
|--------|---------|
| `runs/speedrun.sh` | Full pipeline: train GPT-2 grade model and launch chat UI (~2–3 h on 8×H100) |
| `runs/scaling_laws.sh` | Sweep across model depths to measure scaling laws |
| `runs/miniseries.sh` | Train a series of compute-optimal models |
| `runs/runcpu.sh` | Small model training on CPU/MPS |
| `dev/gen_synthetic_data.py` | Generate synthetic training data for personality/domain injection |
| `dev/repackage_data_reference.py` | Prepare and shard pretraining data |
