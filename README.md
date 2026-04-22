# MLOps Airlines Pipeline

A hands-on exercise for learning MLOps and LLMOps concepts with MLflow.

## What This Repo Contains

Nine short Python exercises that each run independently and show a new concept in the MLflow UI as soon as they finish:

- **MLOps (exercises 1-4)** -- train a flight-delay classifier, register it, promote it, load it for prediction.
- **LLMOps (exercises 5-7b)** -- observe, version, guard, and evaluate LLM calls against an Azure OpenAI deployment.

Every exercise runs against a local MLflow server on port 5000. A `Makefile` wraps each one into a single command.

The dataset is a small Gold-layer sample (`data/gold/`) of US flights, joined across `fact_flights`, `dim_airline`, and `dim_date` -- the same shape produced by a typical dbt pipeline.

---

## Setup

### Option A: Codespaces (recommended, no local install)

1. **Fork this repository**.
2. Click **Code > Codespaces > Create codespace on main**.
3. Wait ~1 minute for Python, MLflow, and dependencies to install.
4. Add your Azure OpenAI API key (needed for exercises 5, 6, 7, 7b):

   ```bash
   cp .env.example .env
   # edit .env and paste AZURE_OPENAI_API_KEY=<your-key>
   ```

### Option B: Local

Requires Python 3.11+.

```bash
pip install uv
uv venv && source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate                # Windows PowerShell
uv sync --active
cp .env.example .env                    # then edit .env
```

### Azure OpenAI configuration

Only the API key lives in `.env`. The non-secret values -- endpoint, deployment name, and API version -- are inlined near the top of every LLMOps exercise file (`exercise-5.py`, `exercise-6.py`, `exercise-7.py`, `exercise-7b.py`):

```python
AZURE_ENDPOINT    = "https://mlops-brisa-trainning-resource.cognitiveservices.azure.com/"
AZURE_API_VERSION = "2024-12-01-preview"
AZURE_DEPLOYMENT  = "gpt-5.4-nano"
```

If your Azure resource uses a different endpoint, deployment, or API version, edit those constants in each file before running.

---

## Running the Exercises

Open two terminals.

**Terminal 1 -- start the MLflow UI:**

```bash
make mlflow
```

Leave this running. On Codespaces, click the forwarded-port notification to open the UI. Locally it is at http://localhost:5000.

**Terminal 2 -- run each exercise:**

### MLOps (~40 min)

| Command | Concept | The one-liner |
|---------|---------|---------------|
| `make exercise-1`  | Tracking          | Params, metrics, and the model artifact -- one source of truth per run |
| `make exercise-2`  | Model Registry    | `@production` is an alias. Deployment stops caring about version numbers |
| `make exercise-3a` | Experimentation   | Sweep many configs, compare them in seconds |
| `make exercise-3b` | Promotion gate    | Promote only if the challenger beats the champion |
| `make exercise-4`  | Serving           | Load by alias. Downstream code never sees a version number |

### LLMOps (~40 min)

| Command | Concept | The one-liner |
|---------|---------|---------------|
| `make exercise-5`  | Observability      | One line of autolog. Every LLM call is an auditable trace |
| `make exercise-6`  | Prompt versioning  | Prompts are code. They have versions. MLflow diffs them |
| `make exercise-7`  | Guardrails         | A guardrail is a system prompt. MLflow logs whether it held |
| `make exercise-7b` | Evaluation         | MLflow runs an LLM judge for you. Guardrail effectiveness = a number |

Each exercise file has a short docstring and one commented `# TODO` block. Uncomment, edit where indicated, run, then check the MLflow UI before moving on.

---

## What to Look For in the MLflow UI

Specific tab and column labels move between MLflow versions, so the table below is observational rather than prescriptive -- open the UI after each run and look for what the exercise produced.

| After | Look for |
|-------|----------|
| `exercise-1`  | One run under your experiment with the params you logged, the metrics, and a saved model artifact |
| `exercise-2`  | A registered model called `FlightDelayClassifier` with a `production` alias; the version it points to matches what your terminal printed |
| `exercise-3a` | Several new runs under `flight-delay-classifier` (one per config); compare them and notice how the F1 metric varies |
| `exercise-3b` | Terminal prints both champion and challenger F1; the `production` alias may now point to a new version |
| `exercise-4`  | Terminal prints `Prediction: DELAYED` or `Prediction: ON TIME` |
| `exercise-5`  | Several captured traces for the experiment; notice how token counts and latency vary with question length |
| `exercise-6`  | A registered prompt named `airAssist-system-prompt` with two versions; comparing them shows the one-word difference |
| `exercise-7`  | Pairs of traces (one bare, one guarded) for each question; the guarded responses follow the refusal rule, the bare ones do not |
| `exercise-7b` | Two evaluation runs (one labelled `bare`, one `guarded`), each logging `stays_on_topic/mean`; the guarded run scores markedly higher |

---

## Project Structure

```
mlops-airlines-pipeline/
|-- .devcontainer/devcontainer.json    # Codespaces setup
|-- data/gold/                         # Gold tables (flights, airlines, dates)
|-- exercises/
|   |-- _utils.py                      # Shared Gold data loader
|   |-- exercise-1.py                  # Tracking
|   |-- exercise-2.py                  # Model Registry
|   |-- exercise-3a.py / exercise-3b.py  # Sweep + promotion gate
|   |-- exercise-4.py                  # Serving
|   |-- exercise-5.py                  # Observability
|   |-- exercise-6.py                  # Prompt versioning
|   `-- exercise-7.py / exercise-7b.py   # Guardrails + evaluation
|-- Makefile                           # `make exercise-N` shortcuts
|-- pyproject.toml                     # Dependencies
|-- .env.example                       # Template for AZURE_OPENAI_API_KEY
`-- README.md
```

---

## Reset

Clear all MLflow runs, models, and prompts between sessions:

```bash
make reset
make mlflow
```

`mlruns/` is gitignored, so this does not affect the repo.

---

## Troubleshooting

### `AZURE_OPENAI_API_KEY` missing or empty
`cat .env` -- it should contain `AZURE_OPENAI_API_KEY=<your-key>` (no quotes, no spaces around `=`). If the line is missing or empty, edit the file and re-run.

### `401 Unauthorized` or `403 Forbidden` on exercises 5, 6, 7, 7b
The key is wrong, has been rotated, or does not have access to the deployment named in the exercise file. Check the key and the `AZURE_DEPLOYMENT` constant against your Azure OpenAI resource in the Azure portal.

### `429 Rate limit` on exercises 5, 6, 7, 7b
The deployment's TPM quota has been hit. Wait a moment and re-run, or raise the quota in the Azure portal under your Azure OpenAI resource.

### `make mlflow` fails on Windows
`mlflow ui` invokes gunicorn, which is Unix-only. Use `mlflow server --host 127.0.0.1 --port 5000` instead.

### Re-running an exercise creates duplicate prompt or model versions
Expected -- MLflow never overwrites. To start fresh, run `make reset`.

### Port 5000 already in use
On Linux or macOS: `lsof -ti :5000 | xargs kill -9` then `make mlflow`.
