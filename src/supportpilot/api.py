"""Thin authenticated HTTP boundary. Run with uvicorn supportpilot.api:app."""

import secrets
import sqlite3
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Annotated
from uuid import UUID

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from supportpilot.common.config import Settings, load_settings
from supportpilot.service.contracts import (
    ChatRequest,
    ChatResult,
    Escalation,
    EscalationUpdate,
    ServiceError,
)
from supportpilot.service.orchestrator import SupportService
from supportpilot.service.runtime import open_service

security = HTTPBearer(auto_error=False)


def create_app(settings: Settings | None = None, service: SupportService | None = None) -> FastAPI:
    cfg = settings or load_settings()
    # Keys are principals, values are secrets, so no token is ever an observable identifier.
    credentials = {k: v.get_secret_value() for k, v in cfg.serving.tokens.items()}
    if any(len(token) < 24 for token in credentials.values()):
        raise ValueError("service tokens must have at least 24 characters")
    if len(set(credentials.values())) != len(credentials):
        raise ValueError("service tokens must be unique")
    if any(k != "operator" and not k.startswith("demo_") for k in credentials):
        raise ValueError("only synthetic demo_ customer principals are allowed")

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        if service is not None:
            application.state.service = service
            yield
        else:
            with open_service(cfg) as runtime:
                application.state.service = runtime
                yield

    application = FastAPI(title="SupportPilot local prototype", lifespan=lifespan)

    def current_service(request: Request) -> SupportService:
        runtime: SupportService | None = getattr(request.app.state, "service", None)
        if runtime is None or runtime.closed:
            raise HTTPException(503, "service_unavailable")
        return runtime

    def principal(auth: Annotated[HTTPAuthorizationCredentials | None, Depends(security)]) -> str:
        if auth is not None:
            for name, token in credentials.items():
                if secrets.compare_digest(auth.credentials.encode(), token.encode()):
                    return name
        raise HTTPException(401, "unauthorized", headers={"WWW-Authenticate": "Bearer"})

    @application.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError) -> JSONResponse:
        return JSONResponse({"detail": exc.code}, status_code=exc.status)

    @application.exception_handler(sqlite3.Error)
    async def storage_error(request: Request, exc: sqlite3.Error) -> JSONResponse:
        return JSONResponse({"detail": "storage_unavailable"}, status_code=503)

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        # Pydantic's default input echo can disclose a submitted token/PII in an error body.
        return JSONResponse({"detail": "invalid_request"}, status_code=422)

    @application.get("/health")
    def health() -> dict[str, str]:
        return {"status": "alive"}

    @application.get("/ready")
    def ready(
        runtime: Annotated[SupportService, Depends(current_service)],
        user: Annotated[str, Depends(principal)],
    ) -> dict[str, str]:
        if not runtime.ready():
            raise HTTPException(503, "service_unavailable")
        return {
            "status": "ready",
            "scope": "authorized_tools_and_safe_fallback",
            "semantic_quality": "NOT_EVALUATED",
        }

    @application.post("/chat", response_model=ChatResult)
    def chat(
        body: ChatRequest,
        runtime: Annotated[SupportService, Depends(current_service)],
        user: Annotated[str, Depends(principal)],
    ) -> ChatResult:
        return runtime.chat(user, body)

    @application.get("/escalations", response_model=list[Escalation])
    def escalations(
        runtime: Annotated[SupportService, Depends(current_service)],
        user: Annotated[str, Depends(principal)],
    ) -> list[Escalation]:
        return runtime.store.escalations(user)

    @application.patch("/escalations/{identity}")
    def transition(
        identity: UUID,
        body: EscalationUpdate,
        runtime: Annotated[SupportService, Depends(current_service)],
        user: Annotated[str, Depends(principal)],
    ) -> dict[str, str]:
        if user != "operator":
            raise HTTPException(403, "operator_required")
        runtime.store.transition(identity, body.status)
        return {"status": body.status}

    return application


app = create_app()
