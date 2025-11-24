from database.models import Domain


async def fetch_all_domains():
    """Fetch all domains from the database"""
    return await Domain.all()


async def fetch_domain_by_id(domain_id: int):
    """Fetch a domain by ID"""
    return await Domain.get_or_none(id=domain_id)


async def create_domain(name: str):
    """Create a new domain"""
    return await Domain.create(name=name)


async def update_domain(domain_id: int, name: str):
    """Update a domain"""
    domain = await Domain.get_or_none(id=domain_id)
    if domain:
        domain.name = name
        await domain.save()
    return domain


async def delete_domain(domain_id: int):
    """Delete a domain"""
    deleted_count = await Domain.filter(id=domain_id).delete()
    return deleted_count > 0
