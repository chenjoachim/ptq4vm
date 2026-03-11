# PTQ4VM — Understanding the Paper and Implementation

Personal notes from reading the codebase. Covers the quantization technique, key design decisions, and practical usage for ViM (Vision Mamba).

---

## What is PTQ4VM?

Post-Training Quantization for Visual Mamba (WACV 2025 Oral). Quantizes pretrained Vim/VMamba models to 4/6/8-bit precision in under 15 minutes with minimal accuracy loss. No retraining of the original model required.

Two separate but parallel codebases exist in this repo:
- **Vim** (Vision Mamba) — root level scripts + `ptq4vm/` + `tools/`
- **VMamba** — `VMamba/classification/` with its own copy of everything

---

## The Core Problem

Activations in Mamba are hard to quantize directly — they have large outliers and a wide dynamic range. Weights are easier. The key idea is to mathematically migrate the quantization difficulty from activations to weights before quantizing either.

---

## Pipeline

### Step 1 — Collect Activation Scales

```bash
python generate_act_scale.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume /path/to/vim_tiny_pretrained.pth \
  --data-path /path/to/imagenet \
  --batch-size 64 \
  --scales-output-path ./act_scales/
```

Registers forward hooks on all `nn.Linear` layers, runs **one batch only**, and records `max(|activation|)` per feature dimension. Saves as `smoothing_t.pt` (tiny), `smoothing_s.pt` (small), `smoothing_b.pt` (base).

**About calibration data**: only one batch is ever used, and labels are never touched — only the images matter. You do not need the full ImageNet training set. The ImageNet validation set works fine and introduces no meaningful data leakage, since nothing is fit to labels.

### Step 2 — JLSS Quantization

```bash
python quant.py \
  --model vim_tiny_patch16_224_bimambav2_final_pool_mean_abs_pos_embed_with_midclstok_div2 \
  --resume /path/to/vim_tiny_pretrained.pth \
  --data-path /path/to/imagenet \
  --act_scales ./act_scales/smoothing_t.pt \
  --qmode ptq4vm \
  --n-lvw 256 --n-lva 256 \
  --alpha 0.5 \
  --epochs 100 \
  --batch-size 64 \
  --train-batch 256
```

Bit-width mapping for `--n-lvw` / `--n-lva`:

| Levels | Bit-width |
|--------|-----------|
| 256    | 8-bit     |
| 64     | 6-bit     |
| 16     | 4-bit     |

---

## Algorithm: JLSS (Joint Learning of Smoothing Scale & Step Size)

Processes the model **one Mamba block at a time** (24 blocks for Vim-Tiny).

### Smoothing Scale (σ)

Computed analytically first (`jlss.py:178`):

```python
scale = (act.pow(alpha) / weight.pow(1 - alpha))
```

Then registered as a learnable parameter and fine-tuned by the optimizer.

Applied as:
- Activations: `x / σ` — suppresses outliers
- Weights: `W × σ` — absorbs the difficulty

Mathematically lossless: `(x/σ) @ (W×σ) = x @ W`. The smoothing itself introduces zero error; only the subsequent rounding does.

`alpha` controls the trade-off. At `alpha=0.5`, difficulty is split evenly. Higher alpha pushes more onto weights.

### Weight Quantization — `Q_Linear` (symmetric)

$$\hat{W} = s_w \cdot \text{clamp}\!\left(\left\lfloor \frac{W \cdot \sigma}{s_w} \right\rceil,\ -q_{max},\ q_{max}\right)$$

- Symmetric around zero
- `s_w` is per-channel (one scalar per output channel)
- `qmax = n_lv/2 - 1` (e.g. 127 for 8-bit)

### Activation Quantization — `Q_Act` (asymmetric)

$$\hat{x} = s_a \cdot \text{clamp}\!\left(\left\lfloor \frac{x/\sigma}{s_a} \right\rceil + z,\ 0,\ n\_lv-1\right) - z \cdot s_a$$

- Asymmetric — has a zero-point `z` to shift the range to wherever activations actually live
- `z = round(-min_val / s_a)` — not learned, recomputed from `s_a`
- `s_a` is learnable (per-token or global)

Activations use asymmetric quantization because they are often skewed (e.g. all-positive after SiLU). Symmetric would waste half the integer range.

### Step Size Initialization

Both `s_w` and `s_a` are initialized via a **grid search over 100 truncation thresholds** — testing candidate max ranges and picking the one that minimizes Lp loss (p=2.4) between original and quantized values. This gives a good starting point before gradient-based fine-tuning.

### Optimization

Only three groups of parameters are trainable — everything else is frozen:

```python
optimizer = AdamW([
    {"params": weight_step_sizes,  "lr": lr_w},   # s_w per Q_Linear
    {"params": act_step_sizes,     "lr": lr_a},   # s_a per Q_Act
    {"params": smooth_scales,      "lr": lr_s},   # σ per layer
])
```

**Loss function** — cosine similarity between FP32 and quantized layer outputs:

```python
loss = (1 - F.cosine_similarity(fp_output, quant_output, dim=-1)).mean()
```

Cosine similarity is used instead of MSE because it focuses on output direction rather than magnitude — more appropriate since downstream LayerNorm normalizes magnitude anyway.

Gradients flow through rounding via the **straight-through estimator** (`RoundQuant`): forward does `round()`, backward passes gradients unchanged.

### Residual Propagation

Mamba blocks return `(output, residual)`. The code tracks `quant_residual` separately alongside `quant_inps`, so each layer is calibrated using the residual stream that a real quantized forward pass would produce — preventing error accumulation mismatches.

---

## Quantized Forward Pass

In simulated quantization mode (default):
1. `Q_Act.forward()`: divide by σ → clamp → round → dequantize
2. `Q_Linear.forward()`: multiply weight by σ → clamp → round → dequantize → `F.linear`

In real int8 mode (`--time_compare`):
- Weights are converted to actual `int8` tensors
- Custom CUDA kernel `vim_GEMM` handles the integer multiply and scale recovery in one fused operation

### Why real int8 is equivalent to simulated quantization

Scales are just scalars — they commute freely with matrix multiplication. For symmetric weights:

$$\hat{X} @ \hat{W} = (s_a \cdot Q_x) @ (s_w \cdot Q_w) = s_a \cdot s_w \cdot (Q_x @ Q_w)$$

So you can do the integer GEMM first, then apply the scales once at the end — identical result, but the multiply-accumulate stays in integer arithmetic the whole time, which is where the hardware speedup comes from.

For asymmetric activations (which have a zero-point $z$), dequantization is $\hat{x} = s_a(q_x - z)$, so the full matrix multiply expands to:

$$\hat{X} @ \hat{W} = s_a \cdot s_w \cdot \left(Q_x @ Q_w - z \cdot \mathbf{1}^T @ Q_w\right)$$

The second term $z \cdot \mathbf{1}^T @ Q_w$ is just a column sum of the weight matrix scaled by $z$ — it does not depend on $x$, so it can be **precomputed once** and subtracted after the GEMM. The kernel effectively does:

```
1. precompute:  col_sum = column sums of int_weight  (done once)
2. integer GEMM: out_int = x_int @ int_weight
3. correct:     out_int -= z * col_sum
4. dequantize:  out = out_int * s_a * s_w
```

Steps 1-3 are pure integer arithmetic. Scales are applied only once at step 4.

---

## Downstream Tasks and Generalization

PTQ4VM quantizes the backbone only — no task-specific head is involved. This means:

- **Same domain, different task**: quantize once, fine-tune only the task head
- **Different image domain**: re-run `generate_act_scale.py` with in-domain images, redo JLSS — activation distributions shift with domain
- **Parameter-efficient fine-tuning**: add LoRA to the FP32 model first, then quantize — cleaner than adding LoRA adapters on top of an already-quantized model
- **QLoRA-style post-hoc adaptation**: architecturally possible but not supported and complicated by the smooth_scale interaction

---

## Key Files

| File | Role |
|------|------|
| `ptq4vm/jlss.py` | JLSS algorithm — main quantization loop |
| `ptq4vm/quantizer.py` | `Q_Linear`, `Q_Act`, initialization, forward passes |
| `generate_act_scale.py` | Activation scale collection (Step 1) |
| `quant.py` | Entry point — runs JLSS and evaluates (Step 2) |
| `tools/models_mamba.py` | Vision Mamba architecture |
| `tools/engine.py` | Evaluation and timing |
| `cuda_measure/vim_GEMM_kernel.cu` | Custom int8 GEMM kernel |
| `VMamba/classification/` | Parallel codebase for VMamba |
