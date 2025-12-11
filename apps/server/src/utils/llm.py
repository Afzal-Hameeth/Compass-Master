import logging
from openai import AzureOpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from typing import List, Dict, Any, Optional
from env import env
import json
import re
import ast

try:
    import yaml  # type: ignore

    _has_yaml = True
except Exception:
    yaml = None
    _has_yaml = False

logger = logging.getLogger(__name__)


def count_tokens(text: str, model: str = "gpt-4") -> int:
    """Count tokens for Azure OpenAI model (rough estimation)"""
    try:
        # Rough estimation: 1 token ≈ 4 characters
        return len(text) // 4
    except Exception as e:
        logger.warning(f"Failed to count tokens: {e}")
        return len(text) // 4


class AzureOpenAIClient:
    def __init__(self):
        self._config = None
        self._client = None
        self.key_vault_url = "https://kvcapabilitycompass.vault.azure.net/"

    def _load_config(self):
        """Load API key from Azure Key Vault, use hardcoded values for other config"""
        if self._config is None:
            kv_url = self.key_vault_url
            api_key = None
            credential = DefaultAzureCredential()
            client = SecretClient(vault_url=kv_url, credential=credential)
            try:
                api_key = client.get_secret("kvCapabilityCompassKeyLLM").value
                logger.info("API Key loaded from Key Vault")
            except Exception as e:
                logger.warning(
                    f"kvCapabilityCompassKeyLLM not found in Key Vault or access denied: {e}; will try environment variable AZURE_OPENAI_API_KEY")
                api_key = env.get("AZURE_OPENAI_API_KEY")

            self._config = {
                "api_key": api_key,
                "endpoint": "https://stg-secureapi.hexaware.com/api/azureai",
                "api_version": "2024-12-01-preview",
                "model": "gpt-4o",
            }

            if self._config.get("api_key"):
                logger.info(f"API Key loaded, starts with: {self._config['api_key'][:5]}...")
                logger.info(
                    f"Azure OpenAI config - Model: {self._config['model']}, Endpoint: {self._config['endpoint']}")

        return self._config

    def _get_client(self):
        if self._client is None:
            config = self._load_config()
            if not config.get("api_key"):
                raise ValueError(
                    "Missing required Azure OpenAI config: api_key. "
                    "Provide this via environment variable AZURE_OPENAI_API_KEY."
                )
            self._client = AzureOpenAI(
                api_key=config["api_key"],
                api_version=config["api_version"],
                azure_endpoint=config["endpoint"]
            )
        return self._client

    async def generate_content(
            self,
            prompt: str,
            context_sections: List[str] = None,
    ) -> Dict[str, Any]:
        # Kept for backward compatibility: small wrapper around the unified generator.
        return await self.generate_json(prompt_text=prompt, purpose="general", context_sections=context_sections)

    async def generate_processes(
            self,
            capability_name: str,
            domain: str,
            process_type: str,
    ) -> Dict[str, Any]:
        """Generate processes for a capability in a specific domain with a given process type using Azure OpenAI LLM"""
        # Use the unified generator with a tight, JSON-only system prompt
        schema_example = {
            "capability_name": capability_name,
            "domain": domain,
            "process_type": process_type,
            "processes": [
                {"name": "Example Process", "description": "",
                 "subprocesses": [{"name": "Example Sub", "description": ""}]}
            ]
        }
        prompt_text = f"For the capability '{capability_name}' in the {domain} domain, generate a list of {process_type}-level processes with their subprocesses. Return the result as a JSON object with the following schema (no markdown, no surrounding text):\n{json.dumps(schema_example, indent=2)}"
        return await self.generate_json(prompt_text=prompt_text, purpose="processes", capability_name=capability_name,
                                        domain=domain, process_type=process_type)

    async def generate_json(self, *, prompt_text: str, purpose: str = "general",
                            context_sections: Optional[List[str]] = None, capability_name: Optional[str] = None,
                            domain: Optional[str] = None, process_type: Optional[str] = None) -> Dict[str, Any]:
        """Unified generator that requests strict JSON and parses robustly using Azure OpenAI.

        - prompt_text: final user-level prompt describing what to generate
        - purpose: optional label (e.g., 'processes') used for logging
        - context_sections: optional list of context snippets to include
        - capability_name: optional capability name for LLM context
        - domain: optional domain name for LLM context
        - process_type: optional process type for LLM context
        """
        try:
            # Import settings manager here to avoid circular imports
            from config.llm_settings import llm_settings_manager
            settings = await llm_settings_manager.get_all_settings()

            # Use vault URL from user-configured settings so secret retrieval
            # fails when the vault is misconfigured (preventing generation)
            vault_url = settings.get("vaultName")
            if vault_url:
                self.key_vault_url = vault_url

            config = self._load_config()
            client = self._get_client()

            
            temperature = settings.get("temperature", 0.2)
            max_tokens = settings.get("maxTokens", 1500)
            top_p = settings.get("topP", 0.9)

            workspace_content = ""
            if context_sections:
                workspace_content += "\n=== CONTENT SECTIONS ===\n"
                for i, section in enumerate(context_sections, 1):
                    workspace_content += f"\n{i}. {section}\n"

            schema_example = {
                "capability_name": capability_name,
                "domain": domain,
                "process_type": process_type,
                "processes": [
                    {"name": "Example Process", "description": "",
                     "subprocesses": [{"name": "Example Sub", "description": ""}]}
                ]
            }

            system_prompt = (
                f"You are an Expert SME in {domain or 'organizational capabilities'} who generates structured process definitions for enterprise capabilities. "
                f"\n\n## Task:\n"
                f"Generate a list of {process_type or 'core'}-level processes for the capability '{capability_name}' within the {domain or 'specified'} domain. "
                f"\n\n## Requirements:\n"
                f"- Generate ONLY two {process_type or 'core'}-level processes relevant to this capability in this domain\n"
                f"- Each process must have a name, description, and list of two subprocesses\n"
                f"- Each subprocess must have a name and description about the subprocess\n"
                f"- Return data as valid JSON matching the provided schema {schema_example}\n"
                f"- Do not invent processes; base them on standard industry practices for {capability_name} in {domain}\n"
                f"- If the capability-domain combination is not recognized, return: {{'error': 'Capability not found for this domain'}}"
            )

            # Generate content using Azure OpenAI
            response = client.chat.completions.create(
                model=config["model"],
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt_text}
                ],
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                frequency_penalty=0.0
            )
            generated = response.choices[0].message.content.strip()

            logger.debug(f"Raw LLM response (first 2000 chars):\n{generated[:2000]}")

            def _clean_candidate(text: str) -> str:
                s = text

                s = re.sub(r"```(?:json|yaml)?\n", "", s)
                s = re.sub(r"```", "", s)
                s = s.replace('`', '')

                s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
                s = re.sub(r"\*([^*]+)\*", r"\1", s)

                match = re.search(r'[{\[]', s)
                if match:
                    start = match.start()
                    s = s[start:]

                try:
                    import html
                    s = html.unescape(s)
                except Exception:
                    pass
                # CRITICAL: Fix double-escaped quotes FIRST: ""key"" -> "key"
                s = re.sub(r'""([^"]*?)""', r'"\1"', s)
                # Normalize all types of quotes to double quotes
                s = s.replace('\u201c', '"').replace('\u201d', '"')  # Smart double quotes
                s = s.replace('\u2018', '"').replace('\u2019', '"')  # Smart single quotes
                s = s.replace('\u201e', '"').replace('\u201f', '"')  # Double low-9 quotes
                s = s.replace('\u2039', '"').replace('\u203a', '"')  # Guillemets
                s = s.replace('«', '"').replace('»', '"')  # French quotes
                # Remove control characters early
                s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
                # Convert remaining single quotes to double quotes
                s = s.replace("'", '"')
                # Fix escaped quotes
                s = s.replace('\\"', '"')
                # Fix double double quotes (again, in case of complex patterns)
                s = re.sub(r'""\s*([^"\n\r]+?)\s*""', r'"\1"', s)
                # Remove trailing commas in objects/arrays
                s = re.sub(r',\s*(?=[}\]])', '', s)
                # Fix missing colons after quoted keys: "key" {value -> "key": {value
                s = re.sub(r'"([^"]+)"\s*([{[])', r'"\1": \2', s)
                # Fix missing colons: "key" "value" -> "key": "value"
                s = re.sub(r'"([^"]+)"\s+"', r'"\1": "', s)
                # Fix unquoted string values (simple heuristic)
                s = re.sub(r': ([A-Za-z][A-Za-z0-9\s&\-]*?)([,}])', r': "\1"\2', s)
                # Handle truncated strings: close any unclosed quoted string at the end
                if s.rstrip().endswith('"') is False and s.rstrip()[-1] not in '}]':
                    # Find the last unclosed quote
                    last_quote = s.rfind('"')
                    if last_quote != -1:
                        after_quote = s[last_quote + 1:].rstrip()
                        if after_quote and not after_quote.startswith(','):
                            s = s[:last_quote + 1] + after_quote.rstrip(',') + '"'
                # Fix incomplete JSON by closing unclosed structures
                open_braces = s.count('{') - s.count('}')
                open_brackets = s.count('[') - s.count(']')
                s = s.rstrip(',').rstrip() + '}' * open_braces + ']' * open_brackets
                return s

            tried = []
            parsed = None

            # 1) Strict JSON
            try:
                parsed = json.loads(generated)
                tried.append('json(strict)')
            except Exception as primary_err:
                tried.append('json(strict) failed')

                # 2) Clean minimally and try json.loads
                candidate = _clean_candidate(generated).strip()
                logger.debug(f"Cleaned candidate (first 2000 chars):\n{candidate[:2000]}")
                try:
                    parsed = json.loads(candidate)
                    tried.append('json(cleaned)')
                except Exception as clean_err:
                    tried.append(f'json(cleaned) failed: {str(clean_err)[:100]}')
                    logger.debug(f"Clean parse failed: {clean_err}")

                # 3) YAML loader
                if parsed is None and _has_yaml:
                    try:
                        parsed = yaml.safe_load(candidate)
                        tried.append('yaml.safe_load')
                    except Exception:
                        tried.append('yaml failed')

                # 4) ast.literal_eval fallback (convert JS literals to Python)
                if parsed is None:
                    try:
                        candidate_py = candidate.replace('true', 'True').replace('false', 'False').replace('null',
                                                                                                           'None')
                        parsed = ast.literal_eval(candidate_py)
                        tried.append('ast.literal_eval')
                        # Normalize list -> wrapped dict when appropriate
                        if isinstance(parsed, (list, tuple)):
                            parsed = {"Core Processes": list(parsed)}
                    except Exception as ast_err:
                        tried.append(f'ast failed: {ast_err}')

                if parsed is None:
                    logger.error(
                        "Failed to parse LLM response. primary_err=%s tried=%s generated_content_snippet=%s",
                        str(primary_err), tried, generated[:2000]
                    )
                    raise ValueError(f"Failed to parse LLM response as JSON: {primary_err}; tried={tried}")

            logger.info(
                f"Generated ({purpose}) for capability '{capability_name or ''}' in domain '{domain or ''}': parsed keys={list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")

            return {"status": "success", "data": parsed, "raw": generated, "capability_name": capability_name}

        except Exception as e:
            logger.error(f"Error generating JSON ({purpose}): {str(e)}")
            raise Exception(f"Process generation failed: {str(e)}")


azure_openai_client = AzureOpenAIClient()
