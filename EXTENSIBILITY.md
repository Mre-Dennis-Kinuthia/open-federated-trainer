# Extensible workloads

This platform now supports three workload classes beyond the original MLP demo.

## From the ops UI

Open **Launch** in the sidebar (or Overview → Open Launch), then **Start from UI**.

That sets the active model, starts train clients on a sample private dataset, starts a
job worker, and enqueues a compute job. Stop processes from the same page.

Requires `ENABLE_LOCAL_LAUNCHER=true` on the coordinator (default on).

## 1. Arbitrary model architectures

Built-ins on the client (`client/src/models/`):

| `model_id` | Description |
|------------|-------------|
| `simple_mlp` | Default federated MLP |
| `tiny_cnn` | Small CNN example |
| `custom` | Load via `MODEL_MODULE=pkg.mod:Class` |

Select for new classic rounds (operator):

```bash
curl -X POST "http://localhost:8000/models/active" \
  -H "Content-Type: application/json" \
  -d '{"model_id":"tiny_cnn","model_config":{"num_epochs":2,"image_size":8}}'
```

Custom plugin example:

```bash
cd client
export PYTHONPATH=src:examples
export MODEL_ID=custom_linear   # or custom + MODEL_MODULE=custom_linear:CustomLinearTrainer
python src/client.py
```

Implement `models.Trainer.train(task, dataset, client_id) -> TrainResult` and call
`register_trainer("my_model", MyTrainer)`.

## 2. Real private datasets

Data **never leaves the client**. Configure:

```bash
export DATASET_PATH=/path/to/data.csv   # or .jsonl / .json / directory / hf:owner/name
export DATASET_FORMAT=auto              # csv | jsonl | json | folder | huggingface
export DATASET_TEXT_COLUMN=text
export DATASET_LABEL_COLUMN=label
```

Try the samples:

```bash
# Text + label (hash-bag features for MLP / CNN)
export DATASET_PATH=examples/sample_private.csv
export DATASET_FORMAT=csv

# Numeric tabular features
export DATASET_PATH=examples/sample_tabular.csv
export DATASET_FORMAT=csv
export DATASET_LABEL_COLUMN=label

python src/client.py
```

LoRA training uses the same loaders via `training/dataset_loader.py`.

## 3. Non-training jobs (general queue)

Job types: `inference`, `label`, `compute` (Folding@home-style work units).

Enqueue:

```bash
# Inference on private inputs (inputs stay on client; only predictions return)
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_type":"inference","payload":{"inputs":["hello edge","federated rocks"],"model_id":"local-scorer"}}'

# Science compute chunk
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_type":"compute","payload":{"formula":"monte_carlo_pi","seed":7,"steps":100000}}'

# Labeling chunk over local dataset
curl -X POST http://localhost:8000/jobs \
  -H "Content-Type: application/json" \
  -d '{"job_type":"label","payload":{"offset":0,"limit":8}}'
```

Run a worker:

```bash
cd client/src
export JOB_TYPES=inference,label,compute
python worker.py
```

List / inspect:

```bash
curl -s http://localhost:8000/jobs | python3 -m json.tool
curl -s http://localhost:8000/models | python3 -m json.tool
```

## Privacy note

Training updates and job **results** are what leave the device. Raw dataset rows,
inference input text (beyond what you put in the job payload), and local files
are not uploaded by these handlers. Prefer `label`/`inference` jobs that read
`DATASET_PATH` locally instead of embedding sensitive text in `payload.inputs`.
