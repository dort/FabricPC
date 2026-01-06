# Multi-GPU Training Fix Plan

## Problem Summary

When running multi-GPU training on a dual GPU system, the loss is reported in **billions** while single-GPU training reports loss of only a **few hundred**.
The Multi-GPU code diverged early in development and is missing local learning dynamics and uses a deprecated initialization config.

## Root Cause Analysis

### Issue 1: State Initialization Config Mismatch (Critical)

The multi-GPU and single-GPU training use different config keys for state initialization:

| File | Line | Config Key Used |
|------|------|-----------------|
| `fabricpc/training/train.py` | 112-113 | `structure.config["graph_state_initializer"]` |
| `fabricpc/training/multi_gpu.py` | 250 | `config.get("state_initialization", None)` |

**Impact:** When the training config lacks a `"state_initialization"` key, multi_gpu.py passes `None` to `initialize_graph_state()`. This causes:
- Different/invalid initial states
- Inference fails to converge
- Energy values explode to billions

### Issue 2: Different Gradient Computation Methods (Critical)

| Aspect | Single GPU | Multi GPU |
|--------|------------|-----------|
| Method | Local Hebbian learning | Autodiff through inference |
| Implementation | `compute_local_weight_gradients()` | `jax.value_and_grad(loss_fn)` |
| Location | `train.py:127` | `multi_gpu.py:155` |

The multi-GPU version uses backpropagation through the entire inference loop, while single-GPU uses local learning rules.

---

## Proposed Fixes

### Fix 1: Align State Initialization Config (Required)

**File:** `fabricpc/training/multi_gpu.py`

**Change line 250 from:**
```python
state_init_config = config.get("state_initialization", None)
```

**To:**
```python
state_init_config = structure.config.get("graph_state_initializer")
```

This ensures multi-GPU training uses the same state initialization as single-GPU.


### Fix 2: Apply Same Fix to Evaluation (Required)

**File:** `fabricpc/training/multi_gpu.py`

Line 348 has the same issue:
```python
state_init_config = config.get("state_initialization", None)
```

Apply the same fix as above.

---

### Fix 3: Hebbian learning in multi-GPU (Required)
- Fully align code with single GPU training by using shared local gradient code
- Shard the gradients across devices
- Use `jax.lax.pmean` to average gradients a


## Files to Modify

| File | Lines | Change |
|------|-------|--------|
| `fabricpc/training/multi_gpu.py` | 250 | Fix state_init_config source |
| `fabricpc/training/multi_gpu.py` | 348 | Fix state_init_config source (eval) |

---

## Testing Plan

1. **Unit test:** Verify `state_init_config` is correctly read from `structure.config`
2. **Integration test:** Run same model on single-GPU and multi-GPU, compare loss values
3. **Regression test:** Ensure existing multi-GPU workflows still function if they were passing config correctly

### Manual Verification

```python
# After fix, these should produce similar loss values:
# Single GPU
params, losses_single, _ = train_pcn(params, structure, loader, config, rng)

# Multi GPU
params_multi = train_pcn_multi_gpu(params, structure, loader, config, rng)
```

Expected: Loss values within same order of magnitude (both ~hundreds, not billions).

---
cross devices before applying updates