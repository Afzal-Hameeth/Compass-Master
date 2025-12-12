from typing import Optional, List
from tortoise.transactions import in_transaction
from tortoise.exceptions import DoesNotExist
from database.models import Capability, Process, Domain


async def create_capability(name: str, description: str, domain_id: Optional[int] = None) -> Capability:
    async with in_transaction():
        domain_obj = None
        if domain_id is not None:
            try:
                domain_obj = await Domain.get(id=domain_id)
            except DoesNotExist:
                domain_obj = None
        cap = await Capability.create(name=name, description=description, domain=domain_obj)
        return cap


async def fetch_all_capabilities() -> List[Capability]:
    return await Capability.filter(deleted_at=None).prefetch_related('processes', 'domain').all()


async def fetch_by_id(capability_id: int) -> Optional[Capability]:
    try:
        # Ensure domain relation is prefetched so `.domain` is a model instance
        return await Capability.filter(id=capability_id, deleted_at=None).prefetch_related('domain').first()
    except DoesNotExist:
        return None


async def update_capability(capability_id: int, name: Optional[str] = None, description: Optional[str] = None, domain_id: Optional[int] = None) -> Optional[Capability]:
    obj = await fetch_by_id(capability_id)
    if not obj:
        return None
    if name is not None:
        obj.name = name
    if description is not None:
        obj.description = description
    if domain_id is not None:
        try:
            dom = await Domain.get(id=domain_id)
            obj.domain = dom
        except DoesNotExist:
            obj.domain = None
    await obj.save()
    return obj


async def delete_capability(capability_id: int) -> bool:
    deleted = await Capability.filter(id=capability_id).delete()
    return deleted > 0
