from fastapi import FastAPI

from app.pipeline import AccessDecisionEngine
from app.schemas import AccessRequest, AccessResponse, HealthResponse


app = FastAPI(
    title="Edge AI Access Control PoC",
    version="1.0.0",
)
decision_engine = AccessDecisionEngine()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.post("/v1/access/verify", response_model=AccessResponse)
def verify_access(request: AccessRequest) -> AccessResponse:
    return decision_engine.verify(request)
