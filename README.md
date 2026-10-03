# Student Pass/Fail MLOps Demo

A beginner project that grows from a small local machine-learning model into an experiment-tracked, packaged, and cloud-hosted prediction service. Follow the stages in order. The first stage is implemented now; later stages are a learning roadmap, not preconfigured services.

> This tiny dataset is for learning only. It is not enough data for a real student outcome or decision.

## Project files

- `train.py` trains a decision tree and saves it as `model.pkl`.
- `predict.py` loads the saved model and predicts the result for a student who studied 7 hours.
- `requirements.txt` lists the packages needed for the first stage.
- `model.pkl` is generated when training runs; it is not source code.

## Before you start

Install Python 3.10 or newer and check that Python and pip are available in a VS Code PowerShell terminal:

```powershell
python --version
python -m pip --version
```

Open this project folder in VS Code. Create and activate a virtual environment, then install the first-stage dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, use the environment's Python directly instead: `\.venv\Scripts\python.exe -m pip install -r requirements.txt`, then run scripts with `\.venv\Scripts\python.exe`.

## Stage 1: train and predict locally

### 1. Understand the example data

Each input is one student's study hours. The matching label is `0` for Fail or `1` for Pass:

| Study hours | Result |
| ---: | --- |
| 1, 2, 3 | Fail (`0`) |
| 4, 5, 6 | Pass (`1`) |

`X` contains the inputs and `y` contains their labels. A decision tree learns a rule from those examples.

### 2. Train and save the model

Run:

```powershell
python train.py
```

Expected output:

```text
Model trained and saved to model.pkl
```

The script calls `model.fit(X, y)` to train and `joblib.dump(...)` to serialize the trained model. The generated `model.pkl` is the model artifact. Run training from this folder so the artifact is saved where `predict.py` expects it.

### 3. Load the model and make a prediction

Run:

```powershell
python predict.py
```

Expected result:

```text
Prediction: [1]
PASS
```

This is inference: `predict.py` loads the existing artifact and predicts without training again. Try changing `study_hours` to `[[2]]`; that example should predict `FAIL`. The decision tree's behavior outside the small training range is not evidence of real-world accuracy.

## Stage 2: track experiments with MLflow

MLflow records which model settings were used, the measured metrics, and model artifacts, so separate training runs can be compared.

1. Install MLflow in the same virtual environment:

   ```powershell
   python -m pip install mlflow
   ```

2. Install the updated requirements (or install MLflow alone with `python -m pip install mlflow`):

   ```powershell
   python -m pip install -r requirements.txt
   ```

3. Run the updated `train.py`:

   ```powershell
   python train.py
   ```

   The script creates/selects the `Student_Pass_Fail` experiment, starts a run, trains a decision tree with `max_depth=2`, and logs the model type and depth as parameters. It predicts the same six rows it trained on, calculates accuracy, logs that metric, and logs the model to MLflow. It also saves `model.pkl` so the existing `predict.py` continues to work. MLflow is configured to trust only the scikit-learn tree type produced by this training script; do not use that trust setting for models from untrusted sources. The reported accuracy is training accuracy, not a trustworthy measure of performance on new students.

4. Run `python train.py` again to create another run. To compare settings, change `max_depth` in `train.py` between runs.
5. Start the local UI in a second terminal, from this project folder and with the same Python environment active:

   ```powershell
   mlflow ui
   ```

   Open <http://127.0.0.1:5000> and inspect the experiment, runs, parameters, metrics, and logged model. With the current local setup, run metadata is stored in `mlflow.db` and model artifacts are stored under `mlruns/`. Stop the UI with Ctrl+C when finished.

## Stage 3: register a model version

Experiment tracking records what happened during training; a model registry gives a selected model a stable name and tracks its versions. After choosing a run, register its logged model as `StudentPassModel` using the MLflow registry workflow supported by your tracking backend. A local file-based store may not support every registry operation; use a registry-capable SQL-backed or hosted tracking store if MLflow asks for one.

Keep the run and version identifiable. When a later run is better, register another version rather than overwriting the old artifact. Do not promote a model based only on training accuracy.

## Stage 4: expose predictions through FastAPI

An API lets another application send input over HTTP instead of running `predict.py` directly.

1. Install the API packages:

   ```powershell
   python -m pip install fastapi uvicorn
   ```

2. Create `app.py` that loads `model.pkl` once when the service starts, defines `GET /` for a health/message response, and defines a prediction route that accepts study hours and returns JSON such as:

   ```json
   {
     "study_hours": 7,
     "prediction": 1,
     "result": "PASS"
   }
   ```

3. Start the local development server:

   ```powershell
   uvicorn app:app --reload
   ```

4. Open <http://127.0.0.1:8000> and try <http://127.0.0.1:8000/predict/7>. Stop the server with Ctrl+C.

For a real service, validate inputs, handle a missing or incompatible model artifact, and protect the endpoint as appropriate.

## Stage 5: package the API with Docker

Docker packages the application and its runtime dependencies into an image, reducing differences between computers.

1. Add `fastapi` and `uvicorn` to `requirements.txt` along with the existing packages.
2. Create a `Dockerfile` that uses a Python base image, copies and installs `requirements.txt`, copies the app and model artifact, exposes port 8000, and starts Uvicorn listening on `0.0.0.0:8000`.
3. With Docker installed and running, build and launch the image:

   ```powershell
   docker build -t student-ml-api .
   docker run --rm -p 8000:8000 student-ml-api
   ```

4. Test <http://localhost:8000/predict/7>. The model artifact must be included in the image or mounted into the container. For repeatable production builds, pin dependency versions and define how the model is delivered.

## Stage 6: move the lifecycle to Azure Machine Learning

Azure ML provides a cloud workspace for jobs, experiments, models, compute, and endpoints. Cloud resources can incur charges; review pricing, region availability, and permissions before creating compute or endpoints.

1. Create or select an Azure subscription, resource group, and Azure Machine Learning workspace. Sign in with an identity that has access to the workspace.
2. Choose suitable compute. Local CPU is enough for this demo; cloud compute is optional and should be stopped or deleted when not needed.
3. Submit `train.py` as an Azure ML job with its code, environment, and inputs. The job runs training remotely and reports its status and logs in the workspace.
4. Track job metrics and artifacts. Azure ML can use MLflow-compatible tracking for experiments.
5. Register the chosen model in the workspace, preserving model name and version information.
6. Deploy the registered model to a managed online endpoint, configure its scoring code and environment, and authorize callers.
7. Send a test request containing `study_hours: 7`; check that the endpoint response reports prediction `1` and result `PASS`. Monitor endpoint health, errors, latency, and prediction quality, and clean up billable resources after practice.

Azure setup is intentionally not automated by this starter: it depends on your subscription, region, permissions, and selected deployment options. Follow the current Azure ML documentation for the workspace and endpoint configuration that applies to your account.

## The complete lifecycle

```text
Data
  -> Train and evaluate
  -> Track experiments with MLflow
  -> Register a selected model version
  -> Package the prediction service with Docker
  -> Deploy to Azure ML
  -> Serve predictions
  -> Monitor the service and model
```

## Troubleshooting

- `ModuleNotFoundError`: activate `.venv` and install dependencies with `python -m pip install -r requirements.txt`.
- `FileNotFoundError: model.pkl`: run `python train.py` from the project folder before `python predict.py`.
- Port 8000 or 5000 is busy: stop the other process or choose another port for the local service/UI.
- Docker cannot connect: start Docker Desktop and wait until its engine is ready.
