import logging
import re
logging.basicConfig(level=logging.INFO)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from tortoise.contrib.pydantic import pydantic_model_creator

from database.repositories import capability_repository, process_repository, domain_repository
from database.models import Capability as CapabilityModel, Process as ProcessModel, Domain as DomainModel
from utils.llm import azure_openai_client

router = APIRouter()
import logging
logger = logging.getLogger(__name__)
class ResearchRequest(BaseModel):
    query: str

@router.post("/capabilities/research")
async def research_capabilities(payload: ResearchRequest):
    """
    Analyze user query using LLM and return relevant capabilities.
    """
    query = payload.query
    # Fetch all capabilities
    capabilities = await capability_repository.fetch_all_capabilities()
    capability_names = [c.name for c in capabilities]

    logger.info(f"[Research] User query: {query}")
    logger.info(f"[Research] Capability names: {capability_names}")

    # Use LLM to analyze query and match capabilities
    llm_result = await azure_openai_client.generate_content(
        prompt=f"Given the following capabilities: {capability_names}. Which are most relevant to the user query: '{query}'? Return a JSON object with a 'capabilities' key containing a list of relevant capability names."
    )
    logger.info(f"[Research] LLM raw result: {llm_result}")

    # Extract relevant names from llm_result['data']['capabilities']
    relevant_names = []
    if isinstance(llm_result, dict):
        data = llm_result.get("data", {})
        if isinstance(data, dict):
            relevant_names = data.get("capabilities", [])
        elif isinstance(data, list):
            relevant_names = data
    logger.info(f"[Research] Relevant capability names from LLM: {relevant_names}")

    relevant_caps = [c for c in capabilities if c.name in relevant_names]
    logger.info(f"[Research] Matched capabilities: {[c.name for c in relevant_caps]}")

    result = [
        {
            "id": c.id,
            "name": c.name,
            "description": c.description,
            "domain": c.domain.name if getattr(c, 'domain', None) else None,
        }
        for c in relevant_caps
    ]
    logger.info(f"[Research] Response payload: {result}")
    return JSONResponse(result)


Domain_Pydantic = pydantic_model_creator(DomainModel, name="Domain")
Capability_Pydantic = pydantic_model_creator(CapabilityModel, name="Capability")
Process_Pydantic = pydantic_model_creator(ProcessModel, name="Process")


class DomainCreateRequest(BaseModel):
    name: str

class CapabilityCreateRequest(BaseModel):
    name: str
    description: str
    process_id: Optional[int] = None
    domain_id: Optional[int] = None


class ProcessCreateRequest(BaseModel):
    name: str
    level: str
    description: str
    capability_id: Optional[int] = None


@router.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)


# CRUD for Domains
@router.post("/domains", response_model=Domain_Pydantic)
async def create_domain(payload: DomainCreateRequest):
    obj = await domain_repository.create_domain(payload.name)
    return await Domain_Pydantic.from_tortoise_orm(obj)


@router.get("/domains", response_model=List[Domain_Pydantic])
async def list_domains():
    domains = await domain_repository.fetch_all_domains()
    return await Domain_Pydantic.from_queryset(DomainModel.all())


@router.get("/domains/{domain_id}", response_model=Domain_Pydantic)
async def get_domain(domain_id: int):
    obj = await domain_repository.fetch_domain_by_id(domain_id)
    if not obj:
        raise HTTPException(status_code=404, detail="Domain not found")
    return await Domain_Pydantic.from_tortoise_orm(obj)


@router.put("/domains/{domain_id}", response_model=Domain_Pydantic)
async def update_domain(domain_id: int, payload: DomainCreateRequest):
    obj = await domain_repository.update_domain(domain_id, payload.name)
    if not obj:
        raise HTTPException(status_code=404, detail="Domain not found")
    return await Domain_Pydantic.from_tortoise_orm(obj)


@router.delete("/domains/{domain_id}")
async def delete_domain(domain_id: int):
    ok = await domain_repository.delete_domain(domain_id)
    if not ok:
        raise HTTPException(status_code=404, detail="Domain not found")
    return {"deleted": True}


@router.post("/capabilities", response_model=Capability_Pydantic)
async def create_capability(payload: CapabilityCreateRequest):
    obj = await capability_repository.create_capability(payload.name, payload.description, payload.domain_id)
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
            "domain": c.domain.name if getattr(c, 'domain', None) else None,
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
    obj = await capability_repository.update_capability(capability_id, name=payload.name, description=payload.description, domain_id=payload.domain_id)
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


class GenerateProcessRequest(BaseModel):
    capability_name: str
    capability_id: int
    domain: str
    process_type: str


@router.post("/processes/generate")
async def generate_processes(payload: GenerateProcessRequest):
    """Generate processes using LLM and save them to the database"""
    try:
        logger.info(f"/processes/generate called with payload: capability_name={payload.capability_name}, capability_id={payload.capability_id}, domain={payload.domain}, process_type={payload.process_type}")
        # Call the LLM to generate processes
        logger.info("Calling azure_openai_client.generate_processes...")
        print(f"[DEBUG] /processes/generate payload: capability_name={payload.capability_name}, capability_id={payload.capability_id}, domain={payload.domain}, process_type={payload.process_type}")
        try:
            llm_result = await azure_openai_client.generate_processes(payload.capability_name, payload.domain, payload.process_type)
            logger.info(f"LLM returned: {llm_result}")
            print(f"[DEBUG] LLM returned: {llm_result}")
        except Exception as e:

            logger.exception("LLM call failed")
            print(f"[DEBUG] LLM call failed: {e}")
            raise HTTPException(status_code=500, detail=f"LLM call failed: {str(e)}")
        
        if llm_result.get("status") != "success":
            raise HTTPException(status_code=500, detail="Failed to generate processes from LLM")
        
        generated_data = llm_result.get("data", {})

        # Normalize different possible keys returned by various LLM prompts/parsers.
        def _extract_core_processes(data):
            # If the LLM returned a list directly, treat it as core processes
            if isinstance(data, list):
                return data
            if not isinstance(data, dict):
                return []

            # Try common variants (case / spacing / snake/camel)
            candidates = [
                "core_processes",
                "coreProcesses",
                "Core Processes",
                "core processes",
                "core-processes",
                "processes",
                "core",
            ]
            for key in candidates:
                if key in data:
                    val = data.get(key)
                    return val if isinstance(val, list) else []

            # Try a case-insensitive, punctuation-insensitive match
            lookup = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in data.items()}
            for target in ("coreprocesses", "coreprocess", "processes"):
                if target in lookup:
                    val = lookup[target]
                    return val if isinstance(val, list) else []

            return []

        core_processes = _extract_core_processes(generated_data)

        if not core_processes:
            # Provide more debug info when parsing failed so frontend can act accordingly
            logger.error("No core processes extracted from LLM result. llm_result keys=%s raw_snippet=%s",
                         list(generated_data.keys()) if isinstance(generated_data, dict) else type(generated_data),
                         llm_result.get('raw', '')[:1000])
            raise HTTPException(status_code=400, detail="No processes were generated")
        
        # Verify capability exists
        capability = await capability_repository.fetch_by_id(payload.capability_id)
        if not capability:
            raise HTTPException(status_code=404, detail="Capability not found")
        
        # Create processes in the database
        created_processes = []
        for core_proc in core_processes:
            # Create the top-level process using the requested process_type
            top_level = payload.process_type or "core"
            core_proc_obj = await process_repository.create_process(
                name=core_proc.get("name", ""),
                level=top_level,
                description=core_proc.get("description", ""),
                capability_id=payload.capability_id,
            )

            created_processes.append({
                "id": core_proc_obj.id,
                "name": core_proc_obj.name,
                "level": core_proc_obj.level,
                "description": core_proc_obj.description,
            })

            # Create subprocesses; subprocesses remain at 'subprocess' level
            subprocesses = core_proc.get("subprocesses", [])
            for subprocess in subprocesses:
                subprocess_obj = await process_repository.create_process(
                    name=subprocess.get("name", ""),
                    level="subprocess",
                    description=subprocess.get("description", ""),
                    capability_id=payload.capability_id,
                )
        
        return {
            "status": "success",
            "message": f"Successfully generated and saved {len(core_processes)} {payload.process_type or 'core'} processes",
            "processes": created_processes,
            "process_type": payload.process_type or 'core',
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating processes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate processes: {str(e)}")


