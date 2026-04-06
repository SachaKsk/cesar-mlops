import os
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Depends, HTTPException

from runtime.inference.load_artifact import load_artifact_from_path, ArtifactNotFoundError
from runtime.inference.estimate_from_artifact import estimate_from_model, InvalidFeatureError
from prediction_contract.request_schema import EstimateRequest
from prediction_contract.response_schema import EstimateResponse
from prediction_contract.contract_version import ContractVersion

app = FastAPI(title="CESAR Prediction API", version="0.1.0")

# Load the model and contract once and reuse for every request. We cache in _loaded so we do not
# read from disk on each call. If CESAR_MODEL_PATH or CESAR_CONTRACT_PATH are missing, we return 503
# (service unavailable) so callers know the API is not ready yet.
_loaded: tuple[object, ContractVersion] | None = None


def get_artifact() -> tuple[object, ContractVersion]:
    global _loaded
    if _loaded is not None:
        return _loaded
    model_path = Path(os.environ.get("CESAR_MODEL_PATH", ""))
    contract_path = Path(os.environ.get("CESAR_CONTRACT_PATH", ""))
    if not model_path or not contract_path:
        raise HTTPException(status_code=503, detail="CESAR_MODEL_PATH and CESAR_CONTRACT_PATH must be set")
    try:
        _loaded = load_artifact_from_path(model_path, contract_path)
        return _loaded
    except ArtifactNotFoundError as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/estimate/", response_model=EstimateResponse)
#this will return the EstimateResponse object with confidence intervals (value_low_eur, value_high_eur) if model supports them, else just point estimate
def post_estimate(
    request: EstimateRequest,
    artifact: Annotated[tuple[object, ContractVersion], Depends(get_artifact)],
) -> EstimateResponse:
    model, contract = artifact
    try:
        return estimate_from_model(model, request, contract)
    except InvalidFeatureError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
