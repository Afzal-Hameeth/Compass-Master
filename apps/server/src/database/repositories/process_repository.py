from typing import Optional, List, Dict, Any
from tortoise.transactions import in_transaction
from tortoise.exceptions import DoesNotExist
from database.models import Process, Capability, SubProcess


async def create_process(name: str, level: str, description: str, capability_id: Optional[int] = None, subprocesses: Optional[List[Dict[str, Any]]] = None, category: Optional[str] = None) -> Process:
    async with in_transaction():
        capability_obj = None
        if capability_id is not None:
            try:
                capability_obj = await Capability.get(id=capability_id)
            except DoesNotExist:
                capability_obj = None
        proc = await Process.create(name=name, level=level, description=description, capability=capability_obj, category=category)
        
        # Create associated subprocesses if provided
        if subprocesses:
            for sub_data in subprocesses:
                await SubProcess.create(
                    name=sub_data.get("name", ""),
                    description=sub_data.get("description", ""),
                    category=sub_data.get("category"),
                    process=proc
                )
        
        return proc


async def list_processes(capability_id: Optional[int] = None) -> List[Process]:
    if capability_id is not None:
        return await Process.filter(deleted_at=None, capability_id=capability_id).all()
    return await Process.filter(deleted_at=None).all()


async def fetch_process_by_id(process_id: int) -> Optional[Process]:
    try:
        return await Process.get(id=process_id, deleted_at=None)
    except DoesNotExist:
        return None


async def update_process(process_id: int, **kwargs) -> Optional[Process]:
    obj = await fetch_process_by_id(process_id)
    if not obj:
        return None
    for k, v in kwargs.items():
        setattr(obj, k, v)
    await obj.save()
    return obj


async def delete_process(process_id: int) -> bool:
    deleted = await Process.filter(id=process_id).delete()
    return deleted > 0
