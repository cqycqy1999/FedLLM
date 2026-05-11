# BP-FedPEFT

Minimal implementation of block-wise progressive federated PEFT for LLM
instruction tuning.

## Code Structure

```text
BP-FedPEFT
├── configs
│   ├── bpfedpeft_sft.yaml
│   ├── bpfedpeft_sft_14clients.yaml
│   └── edge_cluster_14a100.example.json
├── fedpost
│   ├── algorithms        # federated schedules and aggregation
│   ├── bpfedpeft         # block partition and active-parameter utilities
│   ├── data              # dataset loading, processing, and partitioning
│   ├── evaluation        # HumanEval, FinQA, and MedQA evaluation
│   ├── federation        # server, client, sampler, and messages
│   ├── models            # Hugging Face model loading, LoRA, block runtime
│   ├── pipeline          # running interface
│   ├── trainers          # local SFT trainers
│   └── utils             # config, registry, recorder, seed
├── scripts
│   ├── download_paper_assets.py
│   ├── prepare_semantic_dirichlet.py
│   ├── precompute_block_vectors.py
│   ├── run_bpfedpeft.py
│   ├── run_cluster_rounds.py
│   ├── run_edge_worker.py
│   ├── run_paper_eval.py
│   └── summarize_paper_eval.py
├── tests
├── environment.yml
└── requirements.txt
```

## Quick Start

### Step 1. Installation

```bash
conda env create -f environment.yml
conda activate bpfedpeft
```

Or use an existing Python environment:

```bash
pip install -r requirements.txt
```

### Step 2. Prepare CodeAlpaca

The default quick config uses `Qwen/Qwen3-0.6B` and a small CodeAlpaca split.

```bash
python scripts/download_paper_assets.py \
  --asset datasets \
  --only codealpaca_20k \
  --export-jsonl \
  --max-samples 64 \
  --output-dir assets

python scripts/prepare_semantic_dirichlet.py \
  --input assets/jsonl/codealpaca_20k.jsonl \
  --output-dir data/codealpaca_quick_2clients \
  --num-clients 2 \
  --alpha 0.3 \
  --classifier keyword
```

### Step 3. Run BP-FedPEFT

```bash
python scripts/run_bpfedpeft.py --config configs/bpfedpeft_sft.yaml
```

`configs/bpfedpeft_sft.yaml` runs Qwen3-0.6B with LoRA, two simulated clients,
one selected client per round, and six communication rounds.

### Step 4. Run tests

```bash
python -m unittest discover tests
```

## Advanced

### Main configuration fields

```yaml
task: sft
output_dir: outputs/bpfedpeft_qwen3_codealpaca_quick
seed: 42

model:
  model_name_or_path: Qwen/Qwen3-0.6B
  tokenizer_name_or_path: Qwen/Qwen3-0.6B
  trust_remote_code: true
  torch_dtype: bfloat16
  gradient_checkpointing: true

peft:
  method: lora
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]

federated:
  algorithm: bpfedpeft
  num_clients: 2
  clients_per_round: 1
  rounds: 6
  local_epochs: 1
  local_steps: 1

bpfedpeft:
  block_end_layers: [10, 19, 28]
  overlap_layers: 1
  vector_path: null
  anchoring_rounds_per_block: 1
  min_rounds_per_block: 1
  max_rounds_per_block: 1
  use_block_forward: true

data:
  source: local
  data_path: data/codealpaca_quick_2clients
  file_type: jsonl
  prompt_field: prompt
  response_field: response

sft:
  max_length: 256
  lr: 1.0e-4
  batch_size: 1
  grad_accum_steps: 1
```

### Block vectors

Depth vectors are optional. To precompute them:

```bash
python scripts/precompute_block_vectors.py \
  --config configs/bpfedpeft_sft.yaml \
  --output outputs/bpfedpeft_qwen3_codealpaca_quick/depth_vectors.pt
```

Then set `bpfedpeft.vector_path` to the generated file.

## Data

### Instruction data format

Local SFT data can be JSON or JSONL:

```json
{"prompt": "Write a Python function that adds two numbers.", "response": "def add(a, b):\n    return a + b"}
```

For Alpaca-style records with `instruction`, `input`, and `output`, the data
preparation script writes normalized `prompt` and `response` fields.

### Full CodeAlpaca split

```bash
python scripts/download_paper_assets.py \
  --asset datasets \
  --only codealpaca_20k \
  --export-jsonl \
  --output-dir assets

python scripts/prepare_semantic_dirichlet.py \
  --input assets/jsonl/codealpaca_20k.jsonl \
  --output-dir data/semantic_dirichlet_14clients \
  --num-clients 14 \
  --alpha 0.3 \
  --classifier keyword
```

The generated directory can be used as `data.data_path`.

## Evaluation Tools

| Task | Dataset | Metric |
| --- | --- | --- |
| Code generation | HumanEval | `humaneval/pass_at_1` |
| Financial reasoning | FinQA | `finqa/execution_accuracy` |
| Medical QA | MedQA-USMLE | `medqa/accuracy` |

Run evaluation on a merged model:

```bash
python scripts/run_paper_eval.py \
  --config configs/bpfedpeft_sft_14clients.yaml \
  --model-path outputs/bpfedpeft_14clients/exports/round_64/merged_model \
  --tasks humaneval finqa medqa \
  --output-dir outputs/eval_round_64
```

Run evaluation on a base model plus LoRA adapter:

```bash
python scripts/run_paper_eval.py \
  --config configs/bpfedpeft_sft_14clients.yaml \
  --adapter-dir outputs/bpfedpeft_14clients/exports/round_64/adapter_model \
  --tasks humaneval finqa medqa \
  --output-dir outputs/eval_round_64
```

## Distributed Edge Training

`scripts/run_cluster_rounds.py` runs the server on the aggregation machine and
dispatches client updates through SSH and rsync.

```bash
python scripts/run_cluster_rounds.py \
  --config configs/bpfedpeft_sft_14clients.yaml \
  --cluster configs/edge_cluster_14a100.example.json \
  --work-dir outputs/cluster_qwen3 \
  --clients-per-round 8 \
  --dry-run
```

The manifest defines the available clients. The runner adapts
`federated.num_clients` to the manifest size.

## Output Files

```text
outputs/<run_name>/
├── config_snapshot.txt
├── round_metrics.jsonl
├── eval_metrics.jsonl
├── summary.jsonl
├── summary.csv
├── checkpoints/
├── exports/
└── eval_predictions/
```
