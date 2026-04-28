# Finding2 Integration Notes

## Scope

Finding2 tests whether a lightweight synthesizer `f_i` can reconstruct an intermediate residual-stream hidden state `h^(i)` from a small anchor payload, with small reconstruction error and limited downstream impact.

This implementation is staged. The first committed stage is deterministic offline static reconstruction over previously collected residual-stream traces. It does not change the existing federated trainer.

## Existing Inputs Reused

- Residual-stream traces are produced by `fedpost.analysis.hidden_states.collect`.
- Trace format:
  - `hidden_states`: tensor `[num_samples, num_probe_positions, num_layers, hidden_dim]`
  - layer index `0` is embedding output
  - layer index `i > 0` is transformer block output `i`
- Existing full outputs in `outputs/residual_stream/*/hidden_states.pt` can drive Part A without rerunning large model forward passes.

## Modules Added

- `fedpost.analysis.finding2.hooks`
  - `CaptureHiddenStates`
  - supports LLaMA/Qwen/Mistral-style HF causal LM block paths:
    - `model.layers`
    - `transformer.h`
    - `gpt_neox.layers`
    - `transformer.blocks`
    - `decoder.layers`
  - returns `{layer_idx: tensor}`
  - supports `to_cpu=True`
  - uses hooks and can be used under `torch.no_grad()` / `torch.inference_mode()`

- `fedpost.analysis.finding2.synthesizers`
  - abstract `Synthesizer`
  - S0 Exact
  - S1 Cached mean
  - S2 Low-rank PCA/SVD compression
  - S3 Distilled random-feature linear map `h^(0) -> h^(i)`
  - S4 Neighbor diagonal linear regression `h^(i-k) -> h^(i)`
  - S5 Anchor+Correction with low-rank residual plus sparse top-k correction

- `fedpost.analysis.finding2.static_reconstruction`
  - Part A runner over hidden-state traces
  - writes machine-readable JSON and `.pt` results
  - emits warnings for threshold violations

## Determinism Policy

- Default seed is `42`.
- Synthesizers that use randomness construct a local `torch.Generator(device="cpu")`.
- Fitting is CPU-first and deterministic where PyTorch linear algebra permits.
- Smoke tests use synthetic tensors to avoid model or dataset downloads.

## Cost Accounting

Costs are parsed analytically by each synthesizer:

- `cost_flops()`
  - returns an approximate multiply-add count for one hidden vector.
- `cost_params()`
  - returns the number of persistent scalar parameters held by the synthesizer.
- `cost_comm_bytes()`
  - returns bytes in the anchor payload per hidden vector.

Cost formulas are encoded in each synthesizer class docstring and method comments. They are not runtime measurements.

## Part A Static Reconstruction

Protocol implemented now:

1. Load `hidden_states.pt`.
2. Select layer indices, default `{8, 16, 24}`.
3. Split calibration and evaluation tensors deterministically.
4. Fit each synthesizer on calibration traces.
5. Build anchors from evaluation traces.
6. Synthesize `h_hat^(i)`.
7. Compute:
   - relative L2 error `mu = ||h_hat - h|| / ||h||`
   - cosine similarity
8. Save mean/std/count and per-synthesizer costs.

Threshold policy:

- S0 Exact: `mu_mean < 1e-5`, otherwise hard failure when `--fail_on_threshold` is set.
- S5 Anchor+Correction: `mu_mean <= 0.06`, otherwise warning is always recorded.

## Part B Dynamic Downstream

Confirmed semantics:

- Split transformer layers into 4 contiguous blocks.
- Keep all LoRA parameters trainable. Do not freeze LoRA parameters by block.
- During a substituted run, replace only the forward activation at a selected block boundary.
- For boundary layer `i`, inject synthesized `h_hat^(i)` as the input residual stream to the next block.
- Use `h_hat.detach()` by default so gradients do not flow into the synthesizer or anchor construction path.
- LoRA parameters after the substituted boundary receive gradients through the normal forward path.
- LoRA parameters before the substituted boundary remain trainable parameters, but they do not receive gradient from a loss whose forward activation is overwritten at that boundary.
- `w=0` means no overlapping block context and no additional real boundary activations.

Implementation plan:

- Baseline: full LoRA training with standard forward.
- Block-wise: full LoRA training with one selected boundary activation substituted per step or according to a configured boundary schedule.
- The first implementation reports deterministic training/eval loss and saves the adapter. External GSM8K / AlpacaEval quality should be run through the existing evaluation stack after adapter export.
- Claims such as "S5 <= 1pt GSM8K gap" require running the full configured evaluation; quick loss-only smoke tests are not sufficient for the paper conclusion.

## Part C Tolerance Curve

Uses the same activation-substitution path as Part B.

Protocol:

- Inject `h_noisy = h + eps * ||h|| * noise` at selected block boundaries.
- Sweep `eps in {0.0, ..., 0.32}`.
- Evaluate under `--quick` mode with deterministic train/eval loss.
- Full GSM8K / AlpacaEval quality should be run on exported adapters through the existing evaluation stack.
- Fit `mu*` from the observed degradation curve.

## No Silent Failure Rules

- Missing files raise exceptions.
- Missing layers raise exceptions.
- Insufficient calibration/evaluation data raises by default.
- `--allow_small_data` is required for smoke tests on small traces.
- Threshold violations are written to result JSON and printed.
