# Production workloads

The coordinator supports federated training, real local inference and
auto-labeling, and operator-installed scientific compute plugins. Private data
and model execution stay on the worker.

## Start from the UI

Open **Launch** in the sidebar.

- **Demo start** explicitly uses repository sample data.
- **Train clients** accepts a real dataset path under `client/`.
- **Job worker** accepts the same private dataset path for local inference and
  labeling.
- **Jobs** creates real model or allowlisted plugin jobs.

The local launcher is enabled with `ENABLE_LOCAL_LAUNCHER=true`. Disable it on
a public/multi-tenant coordinator and run workers through your orchestrator.

## Pluggable training models

Built-in client trainers:

| `model_id` | Input |
| --- | --- |
| `simple_mlp` | Numeric CSV/JSON columns or deterministic hashed text features |
| `tiny_cnn` | Real images in class subdirectories |
| `custom` | A `models.Trainer` loaded from `MODEL_MODULE` |

Select the architecture for new assignments:

```bash
curl -X POST "http://localhost:8000/models/active" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"simple_mlp","model_config":{"num_epochs":3,"input_dim":4}}'
```

Custom trainer:

```bash
cd client
export PYTHONPATH=src:.
export MODEL_ID=custom
export MODEL_MODULE=examples.custom_linear:CustomLinearTrainer
export DATASET_PATH=examples/sample_tabular.csv
python src/client.py
```

Implement `models.Trainer.train(task, dataset, client_id) -> TrainResult`.

## Private datasets

Real data is required by default:

```bash
export DATASET_PATH=/path/to/client/data/train.csv
export DATASET_FORMAT=auto # csv | jsonl | json | folder | huggingface
export DATASET_TEXT_COLUMN=text
export DATASET_LABEL_COLUMN=label
```

Supported inputs:

- CSV, JSONL, and JSON object rows
- Hugging Face datasets via `hf:owner/name`
- text files in folders
- image classification folders (`root/class_a/*.jpg`,
  `root/class_b/*.jpg`)

If `DATASET_PATH` is absent, the client fails instead of silently training on
fake records. Synthetic data exists only for explicit tests with
`ALLOW_SYNTHETIC_DATA=true`.

The files in `client/examples/` are samples, not production data.

## Real inference

The worker loads an actual Transformers pipeline. Provide a model ID/local
model path and either payload inputs or a local dataset range:

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_type":"inference",
    "payload":{
      "model_id":"distilbert/distilbert-base-uncased-finetuned-sst-2-english",
      "task":"text-classification",
      "offset":0,
      "limit":16
    }
  }'
```

Blank `inputs` means the worker reads its own `DATASET_PATH`. Raw inputs are not
included in the result.

## Real auto-labeling

Use a classification model, or provide candidate labels for a zero-shot model:

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_type":"label",
    "payload":{
      "model_id":"facebook/bart-large-mnli",
      "candidate_labels":["support","billing","technical"],
      "offset":0,
      "limit":16
    }
  }'
```

The returned result contains indices, labels, and confidence scores—not the
private text.

## Scientific compute plugins

Compute jobs require an installed Python entrypoint. Workers execute only
modules in `COMPUTE_PLUGIN_ALLOWLIST`:

```bash
cd client
export PYTHONPATH=src:.
export JOB_TYPES=compute
export COMPUTE_PLUGIN_ALLOWLIST=examples.science_plugin
python src/worker.py
```

Enqueue the included Lennard-Jones molecular-dynamics work unit:

```bash
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{
    "job_type":"compute",
    "payload":{
      "entrypoint":"examples.science_plugin:lennard_jones",
      "work_unit":{
        "positions":[[0,0,0],[1.2,0,0],[0,1.2,0]],
        "steps":250,
        "dt":0.001
      }
    }
  }'
```

Implement additional applications as `function(work_unit: dict) -> JSON value`
and add their module prefix to the worker allowlist.

## LoRA evaluation

Aggregated LoRA adapters are evaluated with the real base model on a coordinator
holdout dataset when configured:

```bash
export LORA_EVAL_DATASET_PATH=/secure/holdout.jsonl
export LORA_EVAL_TEXT_COLUMN=text
export LORA_REQUIRE_EVALUATION=true
export LORA_EVAL_DEVICE=cuda
```

Without an evaluation dataset, evaluation is explicitly marked skipped and no
parameter-norm proxy is reported. With `LORA_REQUIRE_EVALUATION=true`,
aggregation fails until the holdout dataset is configured.

## Operational notes

- The job queue is persisted at `coordinator/data/jobs.json`.
- Workers use claim leases and retry abandoned jobs up to their attempt limit.
- Model downloads may require Hugging Face credentials and substantial disk,
  RAM, or GPU memory.
- `OPERATOR_API_KEY` should be set before exposing job/model/launch mutations.