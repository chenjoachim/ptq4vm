# Plan: Save & Load Quantized Checkpoints

## Goal

Allow the quantized model (after JLSS) to be saved to disk and reloaded for inference — without re-running the 15-minute optimization.

---

## Two modes to support

### Mode A — Simulated quantization (default, no `--time_compare`)
Weights stay in float, quantization is simulated via `Q.Linear` / `Q.Act` forward passes.

### Mode B — Real int8 (with `--time_compare`)
Weights converted to actual `int8` tensors via `set_real_int8()`. Uses custom CUDA kernel for inference.

---

## Changes needed

### 1. `quant.py` — add `--save-quantized` argument
After `JLSS()` completes and before `evaluate()`:
- (Mode B only) call `set_real_int8()` on all `Q.Linear` / `Q.Act` modules
- `torch.save(model.state_dict(), args.save_quantized)`

### 2. `quant.py` — add `--load-quantized` path
New code path that:
1. Builds the model skeleton (same `create_model` + `Q.Linear` replacement block already in `quant.py`)
2. (Mode B only) calls `set_real_int8()` on all `Q.Linear` / `Q.Act` to set correct dtypes before loading
3. `model.load_state_dict(torch.load(args.load_quantized))`
4. Skips `JLSS()` entirely, goes straight to `evaluate()`

### 3. Verify state dict completeness
Confirm that `model.state_dict()` captures:
- `smooth_scale` (σ) in each `Q.Act`
- `step_size` / s_w in each `Q.Linear`
- `step_size` / s_a in each `Q.Act`
- Weight tensors (float for Mode A, int8 for Mode B)

Check by comparing `evaluate()` output between the original run and a save→load round-trip — numbers should be identical.

---

## Open questions

- Does `set_real_int8()` mutate weight tensors in-place or replace them? Need to check `ptq4vm/quantizer.py` to confirm the state dict keys are stable across the call.
- For Mode B, does loading require the CUDA kernel (`vim_GEMM`) to be compiled? If so, document this as a hard dependency for int8 checkpoint loading.
