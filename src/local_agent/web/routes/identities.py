"""Identity management routes."""

from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ...identities import get_identity_manager

router = APIRouter(prefix="/api/identities", tags=["identities"])


class IdentityInfo(BaseModel):
    """Information about an identity."""

    name: str
    is_builtin: bool
    is_active: bool


class ListIdentitiesResponse(BaseModel):
    """Response with list of identities."""

    identities: List[IdentityInfo]
    active: str


class IdentityContentResponse(BaseModel):
    """Response with identity content."""

    name: str
    content: str
    is_builtin: bool


class SetActiveIdentityRequest(BaseModel):
    """Request to set active identity."""

    name: str


@router.get("", response_model=ListIdentitiesResponse)
def list_identities():
    """List all available identities.

    Returns:
        ListIdentitiesResponse with identity list and active identity
    """
    manager = get_identity_manager()
    active = manager.get_active()
    identities = []

    for name in manager.list_identities():
        identities.append(
            IdentityInfo(
                name=name,
                is_builtin=manager.is_builtin(name),
                is_active=(name == active),
            )
        )

    return ListIdentitiesResponse(identities=identities, active=active)


@router.get("/{name}", response_model=IdentityContentResponse)
def get_identity(name: str):
    """Get identity content.

    Args:
        name: Identity name

    Returns:
        IdentityContentResponse with content

    Raises:
        HTTPException: If identity not found (404)
    """
    manager = get_identity_manager()

    try:
        content = manager.get_content(name)
        return IdentityContentResponse(
            name=name,
            content=content,
            is_builtin=manager.is_builtin(name),
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/active")
def set_active_identity(request: SetActiveIdentityRequest):
    """Set the active identity.

    Args:
        request: Request with identity name to activate

    Returns:
        Success status with active identity name

    Raises:
        HTTPException: If identity not found (404)
    """
    manager = get_identity_manager()

    try:
        manager.set_active(request.name)
        return {"success": True, "active": request.name}
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
