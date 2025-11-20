from typing import Optional, List
from tortoise.transactions import in_transaction
from tortoise.exceptions import DoesNotExist
from database.models import Capability, Process


async def create_capability(name: str, description: str) -> Capability:
    async with in_transaction():
        cap = await Capability.create(name=name, description=description)
        return cap


async def fetch_all_capabilities() -> List[Capability]:
    return await Capability.filter(deleted_at=None).prefetch_related('processes').all()


async def fetch_by_id(capability_id: int) -> Optional[Capability]:
    try:
        return await Capability.get(id=capability_id, deleted_at=None)
    except DoesNotExist:
        return None


async def update_capability(capability_id: int, name: Optional[str] = None, description: Optional[str] = None) -> Optional[Capability]:
    obj = await fetch_by_id(capability_id)
    if not obj:
        return None
    if name is not None:
        obj.name = name
    if description is not None:
        obj.description = description
    await obj.save()
    return obj


async def delete_capability(capability_id: int) -> bool:
    deleted = await Capability.filter(id=capability_id).delete()
    return deleted > 0
