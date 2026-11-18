# MLflow tracking stores

- `gpu_eval.db`: the Kaggle GPU evaluation (`notebooks/kaggle_gpu_eval.ipynb`).
  Experiment `civil-code-rag-chunking`: 5 chunking configs on the CI subset plus
  the full 54-question run. Registered model `civil-code-rag-chunking` v1, alias `@production`.
- `quantization_and_drift.db`: the AWQ vs fp16 comparison (2 runs) and the
  query-drift check (6 runs).
- `mlruns/`: the run artifacts from the Kaggle evaluation.

Open one with:

    mlflow ui --backend-store-uri sqlite:///reports/mlflow/gpu_eval.db

Params, metrics, runs and the registry open directly. Artifact paths were
recorded on Kaggle (/kaggle/working/...), so the Artifacts tab won't load them;
the files themselves are in mlruns/.
