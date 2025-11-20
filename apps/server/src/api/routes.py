import logging
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from tortoise.contrib.pydantic import pydantic_model_creator

from database.repositories import capability_repository, process_repository
from database.models import Capability as CapabilityModel, Process as ProcessModel

router = APIRouter()
logger = logging.getLogger(__name__)


Capability_Pydantic = pydantic_model_creator(CapabilityModel, name="Capability")
Process_Pydantic = pydantic_model_creator(ProcessModel, name="Process")


class CapabilityCreateRequest(BaseModel):
    name: str
    description: str
    process_id: Optional[int] = None


class ProcessCreateRequest(BaseModel):
    name: str
    level: str
    description: str
    capability_id: Optional[int] = None


@router.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)


@router.post("/capabilities", response_model=Capability_Pydantic)
async def create_capability(payload: CapabilityCreateRequest):
    obj = await capability_repository.create_capability(payload.name, payload.description)
    return await Capability_Pydantic.from_tortoise_orm(obj)


@router.get("/capabilities")
async def list_capabilities():
    caps = await capability_repository.fetch_all_capabilities()
    result = []
    for c in caps:

        try:
            procs = await c.processes.all()
        except Exception:
            procs = []

        proc_list = []
        for p in procs:

            level = getattr(p.level, 'value', p.level)
            proc_list.append({
                "id": p.id,
                "name": p.name,
                "level": level,
                "description": p.description,
            })

        result.append({
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "processes": proc_list,
        })

    return JSONResponse(result)


@router.get("/capabilities/{capability_id}", response_model=Capability_Pydantic)
async def get_capability(capability_id: int):
    obj = await capability_repository.fetch_by_id(capability_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Capability not found")
    return await Capability_Pydantic.from_tortoise_orm(obj)


@router.put("/capabilities/{capability_id}", response_model=Capability_Pydantic)
async def update_capability(capability_id: int, payload: CapabilityCreateRequest):
    obj = await capability_repository.update_capability(capability_id, name=payload.name, description=payload.description)
    if not obj:
        raise HTTPException(status_code=404, detail="Capability not found")
    return await Capability_Pydantic.from_tortoise_orm(obj)


@router.delete("/capabilities/{capability_id}")
async def delete_capability(capability_id: int):
    ok = await capability_repository.delete_capability(capability_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Capability not found")
    return {"deleted": True}


@router.post("/processes", response_model=Process_Pydantic)
async def create_process(payload: ProcessCreateRequest):
    obj = await process_repository.create_process(payload.name, payload.level, payload.description, payload.capability_id)
    return await Process_Pydantic.from_tortoise_orm(obj)


@router.get("/processes", response_model=List[Process_Pydantic])
async def list_processes(capability_id: Optional[int] = Query(None, alias="capability_id")):

    prots = await process_repository.list_processes(capability_id)

    try:
        return await Process_Pydantic.from_queryset(ProcessModel.filter(deleted_at=None) if capability_id is None else ProcessModel.filter(deleted_at=None, capability_id=capability_id))
    except Exception:
        return [await Process_Pydantic.from_tortoise_orm(p) for p in prots]


@router.get("/processes/{process_id}", response_model=Process_Pydantic)
async def get_process(process_id: int):
    obj = await process_repository.fetch_process_by_id(process_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Process not found")
    return await Process_Pydantic.from_tortoise_orm(obj)


@router.put("/processes/{process_id}", response_model=Process_Pydantic)
async def update_process(process_id: int, payload: ProcessCreateRequest):
    obj = await process_repository.update_process(process_id, name=payload.name, level=payload.level, description=payload.description)
    if not obj:
        raise HTTPException(status_code=404, detail="Process not found")
    return await Process_Pydantic.from_tortoise_orm(obj)


@router.delete("/processes/{process_id}")
async def delete_process(process_id: int):
    ok = await process_repository.delete_process(process_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Process not found")
    return {"deleted": True}


