# FedLLM Repository Notes

## 1. Model Loading Entry Point

- Main entry point: `fedpost.models.loader.HFModelManager`.
- `HFModelManager.build()` creates tokenizer, model, optional PEFT wrapper, optional DPO reference model, and a `ModelStateSpec`.
- Tokenizer loading uses `AutoTokenizer.from_pretrained(...)`.
- Model loading uses `AutoModelForCausalLM.from_pretrained(...)`.
- Relevant model config fields:
  - `model.model_name_or_path`
  - `model.tokenizer_name_or_path`
  - `model.trust_remote_code`
  - `model.torch_dtype`
  - `model.use_flash_attn`
  - `model.gradient_checkpointing`
- FlashAttention is enabled by passing `attn_implementation="flash_attention_2"` when `model.use_flash_attn=true`.
- Gradient checkpointing is enabled through `model.gradient_checkpointing_enable()` and `model.config.use_cache=false`.

## 2. PEFT Mounting, LoRA-Focused

- PEFT logic lives in `fedpost.models.peft_utils`.
- Current supported PEFT methods are:
  - `none`
  - `lora`
- LoRA mounting entry point: `apply_lora(model, peft_cfg)`.
- LoRA uses Hugging Face PEFT:
  - `LoraConfig`
  - `get_peft_model`
  - `get_peft_model_state_dict`
  - `set_peft_model_state_dict`
- LoRA config fields:
  - `peft.r`
  - `peft.alpha`
  - `peft.dropout`
  - `peft.target_modules`
  - `peft.adapter_name`
- `validate_lora_targets(...)` checks that configured target module name fragments exist in `model.named_modules()`.
- Trainable state export for LoRA uses `export_peft_state(...)`, and tensors are detached, cloned, and moved to CPU.
- Trainable state load for LoRA uses `load_peft_state(...)`.
- Artifact export for LoRA can save:
  - raw adapter state: `adapter_state.pt`
  - PEFT adapter directory: `adapter_model/`
  - optional merged HF model: `merged_model/`

## 3. Federated Training Loop and Checkpoint Naming

- Main loop: `fedpost.federation.coordinator.Coordinator.train()`.
- Per-round flow in `Coordinator.run_round(round_idx)`:
  1. Set `server.round_idx`.
  2. Sample clients through `UniformClientSampler`.
  3. Run algorithm hook `before_broadcast(...)`.
  4. Create payload through `algorithm.make_broadcast_payload(...)`.
  5. Run selected clients in batches through `client_executor.run_batch(...)`.
  6. Run algorithm hook `after_local_train(...)`.
  7. Validate success rate before aggregation.
  8. Apply server update through `algorithm.server_update(...)`.
  9. Export adapter/merged artifacts as configured.
  10. Optionally evaluate.
  11. Record round metrics and summary.
- Client local train entry:
  - `fedpost.federation.client.Client.run_round(...)`
  - `fedpost.federation.client.Client.local_train(...)`
- Server update entry:
  - `fedpost.federation.server.Server.apply_updates(...)`
- Checkpoint naming:
  - `Coordinator._ckpt_path(round_idx)` writes to:
    - `<output_dir>/checkpoints/round_{round_idx + 1}.pt`
- Export artifact naming:
  - `Server.export_round_artifacts(round_idx, ...)` writes to:
    - `<output_dir>/exports/round_{round_idx + 1}/`
  - LoRA adapter state:
    - `<output_dir>/exports/round_{round_idx + 1}/adapter_state.pt`
  - LoRA adapter directory:
    - `<output_dir>/exports/round_{round_idx + 1}/adapter_model/`
  - Optional merged model:
    - `<output_dir>/exports/round_{round_idx + 1}/merged_model/`
- Metrics files written by `Recorder`:
  - `<output_dir>/round_metrics.jsonl`
  - `<output_dir>/eval_metrics.jsonl`
  - `<output_dir>/summary.jsonl`
  - `<output_dir>/summary.csv`
  - `<output_dir>/best_round.json`

## 4. Supported Base Models and Datasets

### Base Models

- The code path is generic for Hugging Face causal language models through `AutoModelForCausalLM`.
- There is no explicit model allowlist in code.
- Existing configs use GPT-2 examples, especially `model_name_or_path: gpt2`.
- Required backbones for the residual-stream experiments:
  - `meta-llama/Meta-Llama-3-8B`
  - `Qwen/Qwen2-7B`
  - `mistralai/Mistral-7B-v0.1`
- These should be loadable through the generic HF path if dependencies, model access, memory, and `trust_remote_code` settings are correct.
- Missing for experiments:
  - No dedicated config files for Llama-3-8B, Qwen2-7B, or Mistral-7B.
  - No analysis-specific model loader wrapper for hidden-state capture yet.

### Training Dataset Adapters

- Dataset adapter registry imports live in `fedpost.data.adapters.__init__`.
- Currently registered adapters:
  - `databricks/databricks-dolly-15k` for SFT.
  - `HuggingFaceH4/ultrafeedback_binarized` for DPO.
  - `Anthropic/hh-rlhf` for DPO.
- HF dataset loading currently uses `datasets.load_dataset(dataset_name, split=dataset_split)` through `HFDatasetLoader`.

### Probe Dataset Requirements

- Required probe data for residual-stream experiments:
  - Alpaca
  - GSM8K
  - MMLU
- Current training data adapters do not include Alpaca, GSM8K, or MMLU probe loaders.
- Needed for the residual-stream experiment:
  - Add an analysis-side probe data loader that can produce plain text prompts from Alpaca, GSM8K, and MMLU without depending on the federated training sample classes.
  - The analysis loader should be separate from `fedpost.data.adapters` because residual-stream probes do not need SFT/DPO formatting or federated partitioning.

## 5. Implementation Implications for Residual-Stream Experiments

- Hidden-state collection should use a separate analysis entry point rather than the federated training coordinator.
- It can reuse HF model/tokenizer conventions, but should avoid mutating training logic.
- LoRA is currently supported as the only PEFT method that can be saved/loaded as adapter state.
- Probe data support must be added for Alpaca/GSM8K/MMLU.
- Large backbone runs will require `device_map="auto"` or an equivalent accelerate-backed path.
