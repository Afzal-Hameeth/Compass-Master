from typing import Optional, Dict, Any
from tortoise.exceptions import DoesNotExist
from database.models import LLMSettings


async def get_settings() -> Optional[LLMSettings]:
    """Fetch the current LLM settings"""
    try:
        # There should only be one settings record, fetch the first one
        return await LLMSettings.first()
    except DoesNotExist:
        return None


async def update_settings(settings_data: Dict[str, Any]) -> LLMSettings:
    """Update or create LLM settings"""
    # Get existing settings or create new one
    settings = await LLMSettings.first()
    
    if settings:
        # Update existing record
        for key, value in settings_data.items():
            setattr(settings, key, value)
        await settings.save()
    else:
        # Create new record
        settings = await LLMSettings.create(**settings_data)
    
    return settings


async def get_all_settings_dict() -> Dict[str, Any]:
    """Get all settings as a dictionary"""
    settings = await get_settings()
    if settings:
        return {
            "provider": settings.provider,
            "vaultName": settings.vault_name,
            "temperature": settings.temperature,
            "maxTokens": settings.max_tokens,
            "topP": settings.top_p,
        }
    # Return defaults if no settings exist
    return {
        "provider": "secure",
        "vaultName": "https://kvcapabilitycompass.vault.azure.net/",
        "temperature": 0.2,
        "maxTokens": 1500,
        "topP": 0.9,
    }
