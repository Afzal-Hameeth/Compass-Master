"""
LLM Settings Management Module

Handles persistent storage and retrieval of LLM configuration settings from database
"""
from typing import Dict, Any
import logging
from database.repositories import llm_settings_repository

logger = logging.getLogger(__name__)

# Default settings
DEFAULT_SETTINGS = {
    "provider": "secure",
    "vaultName": "https://kvcapabilitycompass.vault.azure.net/",
    "temperature": 0.2,
    "maxTokens": 1500,
    "topP": 0.9,
}


class LLMSettingsManager:
    """Manages LLM configuration settings with database persistence"""

    async def get_all_settings(self) -> Dict[str, Any]:
        """Get all current settings from database"""
        return await llm_settings_repository.get_all_settings_dict()

    async def get_setting(self, key: str, default: Any = None) -> Any:
        """Get a specific setting from database"""
        settings = await self.get_all_settings()
        return settings.get(key, default)

    async def update_settings(self, new_settings: Dict[str, Any]) -> Dict[str, Any]:
        """Update settings and persist to database"""
        # Map frontend keys to database field names
        db_data = {}
        if "provider" in new_settings:
            db_data["provider"] = new_settings["provider"]
        if "vaultName" in new_settings:
            db_data["vault_name"] = new_settings["vaultName"]
        if "temperature" in new_settings:
            db_data["temperature"] = new_settings["temperature"]
        if "maxTokens" in new_settings:
            db_data["max_tokens"] = new_settings["maxTokens"]
        if "topP" in new_settings:
            db_data["top_p"] = new_settings["topP"]
        
        if db_data:
            await llm_settings_repository.update_settings(db_data)
        
        return await self.get_all_settings()

    async def reset_to_defaults(self) -> Dict[str, Any]:
        """Reset all settings to defaults"""
        await llm_settings_repository.update_settings(DEFAULT_SETTINGS)
        return await self.get_all_settings()


# Global instance
llm_settings_manager = LLMSettingsManager()
