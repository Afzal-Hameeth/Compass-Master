import logging
import re
logging.basicConfig(level=logging.INFO)
from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
from tortoise.contrib.pydantic import pydantic_model_creator

from database.repositories import capability_repository, process_repository, domain_repository
from database.models import Capability as CapabilityModel, Process as ProcessModel, Domain as DomainModel, SubProcess as SubProcessModel
from utils.llm import azure_openai_client
from utils.llm2 import gemini_client
from config.llm_settings import llm_settings_manager

router = APIRouter()
logger = logging.getLogger(__name__)

class LLMProviderRequest(BaseModel):
    provider: str  # "azure", "gemini", or "secure"

class LLMConfigRequest(BaseModel):
    provider: str
    vaultName: str
    temperature: float
    maxTokens: int
    topP: float

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
    
    provider = await llm_settings_manager.get_setting("provider", "secure")
    logger.info(f"[Research] Using LLM provider: {provider}")

    # Select LLM client based on current provider
    if provider == "gemini":
        llm_client = gemini_client
    else:
        llm_client = azure_openai_client

    # Use LLM to analyze query and match capabilities
    llm_result = await llm_client.generate_content(
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


class SubProcessCreateRequest(BaseModel):
    name: str
    description: Optional[str] = None

class ProcessCreateRequest(BaseModel):
    name: str
    level: str
    description: str
    capability_id: Optional[int] = None
    subprocesses: Optional[List[SubProcessCreateRequest]] = None


@router.get("/health")
async def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)


# LLM Provider Management
@router.get("/settings/llm-provider")
async def get_llm_provider():
    """Get current LLM provider and configuration"""
    settings = await llm_settings_manager.get_all_settings()
    return JSONResponse(settings)


@router.post("/settings/llm-provider")
async def set_llm_provider(payload: LLMProviderRequest):
    """Set the LLM provider (azure, gemini, or secure)"""
    if payload.provider not in ["azure", "gemini", "secure"]:
        raise HTTPException(status_code=400, detail="Invalid provider. Must be 'azure', 'gemini', or 'secure'")
    settings = await llm_settings_manager.update_settings({"provider": payload.provider})
    logger.info(f"LLM provider changed to: {payload.provider}")
    return JSONResponse({"provider": payload.provider, "message": f"Switched to {payload.provider} LLM"})


@router.post("/settings/llm-config")
async def set_llm_config(payload: LLMConfigRequest):
    """Set the LLM configuration"""
    if payload.provider not in ["azure", "gemini", "secure"]:
        raise HTTPException(status_code=400, detail="Invalid provider. Must be 'azure', 'gemini', or 'secure'")
    
    if not (0 <= payload.temperature <= 1):
        raise HTTPException(status_code=400, detail="Temperature must be between 0 and 1")
    
    if not (256 <= payload.maxTokens <= 4096):
        raise HTTPException(status_code=400, detail="Max tokens must be between 256 and 4096")
    
    if not (0 <= payload.topP <= 1):
        raise HTTPException(status_code=400, detail="Top P must be between 0 and 1")
    
    new_settings = {
        "provider": payload.provider,
        "vaultName": payload.vaultName,
        "temperature": payload.temperature,
        "maxTokens": payload.maxTokens,
        "topP": payload.topP,
    }
    settings = await llm_settings_manager.update_settings(new_settings)
    logger.info(f"LLM configuration updated: {new_settings}")
    
    return JSONResponse({
        "status": "success",
        "message": "LLM configuration updated successfully",
        **settings
    })


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
            
            # Fetch subprocesses for this process
            try:
                subprocs = await p.subprocesses.all()
            except Exception:
                subprocs = []
            
            subprocess_list = [
                {
                    "id": sp.id,
                    "name": sp.name,
                    "description": sp.description,
                }
                for sp in subprocs
            ]
            
            proc_list.append({
                "id": p.id,
                "name": p.name,
                "level": level,
                "description": p.description,
                "subprocesses": subprocess_list,
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
    subprocesses_data = []
    if payload.subprocesses:
        subprocesses_data = [
            {"name": sp.name, "description": sp.description or ""}
            for sp in payload.subprocesses
        ]
    obj = await process_repository.create_process(
        payload.name,
        payload.level,
        payload.description,
        payload.capability_id,
        subprocesses=subprocesses_data
    )
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
        
        provider = await llm_settings_manager.get_setting("provider", "secure")
        logger.info(f"Using LLM provider: {provider}")
        
        # Select LLM client based on current provider
        if provider == "gemini":
            llm_client = gemini_client
        else:
            llm_client = azure_openai_client
        
        # Call the LLM to generate processes
        logger.info(f"Calling {provider} LLM client.generate_processes...")
        print(f"[DEBUG] /processes/generate payload: capability_name={payload.capability_name}, capability_id={payload.capability_id}, domain={payload.domain}, process_type={payload.process_type}")
        try:
            llm_result = await llm_client.generate_processes(payload.capability_name, payload.domain, payload.process_type)
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
        
        # Verify capability exists (just for validation; we don't persist yet)
        capability = await capability_repository.fetch_by_id(payload.capability_id)
        if not capability:
            raise HTTPException(status_code=404, detail="Capability not found")
        
        # Return the LLM-generated data WITHOUT persisting (user must approve first)
        # Frontend will show this data in a preview modal with checkboxes, then call a separate endpoint to create selected items
        
        return {
            "status": "success",
            "message": f"Generated {len(core_processes)} {payload.process_type or 'core'} processes (preview only)",
            "processes": [],
            "data": {
                "core_processes": core_processes,
            },
            "process_type": payload.process_type or 'core',
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error generating processes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to generate processes: {str(e)}")


