import logging
from openai import AzureOpenAI, OpenAI
from azure.identity import DefaultAzureCredential
from azure.keyvault.secrets import SecretClient
from typing import List, Dict, Any, Optional
import tiktoken
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

def count_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
  try:
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))
  except Exception as e:
    logger.warning(f"Failed to count tokens with tiktoken: {e}")
    # Fallback: rough estimation (1 token ≈ 4 characters)
    return len(text) // 4


class AzureOpenAIClient:
    def __init__(self):
        self.key_vault_url = "https://kv-fs-to-autogen.vault.azure.net/" or "https://KV-fs-to-autogen.vault.azure.net/"
        self._config = None
        self._client = None

    def _load_config_from_vault(self):
        """Load config from Azure Key Vault"""
        if self._config is None:
            kv_url = env.get("KEY_VAULT_URL") or self.key_vault_url
            if not kv_url:
                logger.info("No Key Vault URL configured; using environment variables for Azure OpenAI config")
                self._config = {
                    "api_key": env.get("AzureLLMKey", "5126416ec6417f51-Fs-Capability-Compass"),
                    "api_base": env.get("AzureOpenAiBase", "https://stg-secureapi.hexaware.com/api/azureai"),
                    "model_version": "2024-02-01",
                    "deployment": env.get("AzureOpenAiDeployment", "gpt-4.1"),
                }
            else:
                try:
                    credential = DefaultAzureCredential()
                    client = SecretClient(vault_url=kv_url, credential=credential)
                    cfg = {}
                    try:
                        cfg["api_key"] = client.get_secret("AzureLLMKey").value
                    except Exception:
                        logger.warning("AzureLLMKey not found in Key Vault or access denied; will try environment variable AZURE_OPENAI_API_KEY")
                        cfg["api_key"] = None
                    try:
                        cfg["api_base"] = client.get_secret("AzureOpenAiBase").value
                    except Exception:
                        logger.warning("AzureOpenAiBase not found in Key Vault or access denied; will try environment variable AZURE_OPENAI_ENDPOINT")
                        cfg["api_base"] = None
                    try:
                        cfg["model_version"] = client.get_secret("AzureOpenAiVersion").value
                    except Exception:
                        logger.warning("AzureOpenAiVersion not found in Key Vault or access denied; will try environment variable AZURE_OPENAI_API_VERSION")
                        cfg["model_version"] = None
                    try:
                        cfg["deployment"] = client.get_secret("AzureOpenAiDeployment").value
                    except Exception:
                        logger.warning("AzureOpenAiDeployment not found in Key Vault or access denied; will try environment variable AZURE_OPENAI_DEPLOYMENT")
                        cfg["deployment"] = None
                    # Fill any missing values from environment variables
                    cfg["api_key"] = cfg.get("api_key") or env.get("AZURE_OPENAI_API_KEY")
                    cfg["api_base"] = cfg.get("api_base") or env.get("AZURE_OPENAI_ENDPOINT")
                    cfg["model_version"] = cfg.get("model_version") or env.get("AZURE_OPENAI_API_VERSION", "2024-02-01")
                    cfg["deployment"] = cfg.get("deployment") or env.get("AZURE_OPENAI_DEPLOYMENT")

                    self._config = cfg

                    if self._config.get("api_base"):
                        logger.info(f"Loaded Azure OpenAI config - Base: {self._config['api_base']}")
                    if self._config.get("model_version"):
                        logger.info(f"Model version: {self._config['model_version']}")
                    if self._config.get("deployment"):
                        logger.info(f"Deployment: {self._config['deployment']}")
                    if self._config.get("api_key"):
                        logger.info(f"API Key starts with: {self._config['api_key'][:5]}...")

                except Exception as e:
                    logger.warning(f"Failed to access Azure Key Vault at {kv_url}: {str(e)}; falling back to environment variables")
                    self._config = {
                        "api_key": env.get("AZURE_OPENAI_API_KEY"),
                        "api_base": env.get("AZURE_OPENAI_ENDPOINT"),
                        "model_version": env.get("AZURE_OPENAI_API_VERSION", "2024-02-01"),
                        "deployment": env.get("AZURE_OPENAI_DEPLOYMENT"),
                    }
        return self._config

    def _get_client(self):
        if self._client is None:
            config = self._load_config_from_vault()
            missing = [k for k in ("api_key", "api_base", "deployment") if not config.get(k)]
            if missing:
                # Provide detailed guidance to help operators fix environment
                raise ValueError(
                    "Missing required Azure OpenAI config: {}. "
                    "Provide these via environment variables (AZURE_OPENAI_API_KEY, AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_DEPLOYMENT) "
                    "or configure them in Azure Key Vault and set KEY_VAULT_URL.".format(
                        ", ".join(missing)
                    )
                )
            self._client = AzureOpenAI(
                azure_endpoint=config["api_base"],
                api_key=config["api_key"],
                api_version=config["model_version"],
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
        process_name: str,
        domain: str,
        process_type: str,
    ) -> Dict[str, Any]:
        """Generate processes based on a process name, domain, and process type using LLM"""
        # Use the unified generator with a tight, JSON-only system prompt and a small wrapper
        schema_example = {
            "process_name": process_name,
            "core_processes": [
                {"name": "Example Core", "description": "", "subprocesses": [{"name": "Example Sub", "lifecycle_phase": "Execution"}]}
            ]
        }
        prompt_text = f"For the {domain} domain, capability '{process_name}' and process type '{process_type}', generate a JSON object with the following schema exactly (no markdown, no surrounding text):\n{json.dumps(schema_example, indent=2)}"
        return await self.generate_json(prompt_text=prompt_text, purpose="processes", process_name=process_name, domain=domain, process_type=process_type)

    async def generate_json(self, *, prompt_text: str, purpose: str = "general", context_sections: Optional[List[str]] = None, process_name: Optional[str] = None, domain: Optional[str] = None, process_type: Optional[str] = None) -> Dict[str, Any]:
        """Unified generator that requests strict JSON and parses robustly.

        - prompt_text: final user-level prompt describing what to generate
        - purpose: optional label (e.g., 'processes') used for logging
        - context_sections: optional list of context snippets to include
        - domain: optional domain name for LLM context
        - process_type: optional process type for LLM context
        """
        try:
            config = self._load_config_from_vault()
            client = self._get_client()

            workspace_content = ""
            if context_sections:
                workspace_content += "\n=== CONTENT SECTIONS ===\n"
                for i, section in enumerate(context_sections, 1):
                    workspace_content += f"\n{i}. {section}\n"

            # Strong system prompt enforcing JSON-only output with a schema example
            system_prompt = (
                f"You are an Expert SME in {domain or 'organizational capabilities'} who answers user queries about organizational capabilities, processes, and subprocesses. "
                f"Your task is to return responses in a structured process-definition manner based on the capability requested by the user. "
                f"The user will provide a capability name, domain, and process type, and you must return the relevant processes and subprocesses exactly in the defined style. "
                f"\n\n### Rules:\n"
                f"- Always respond with the capability name, followed by its processes and subprocesses.\n"
                f"- Generate processes aligned with the process type: {process_type or 'core'}.\n"
                f"- Each process must list its subprocesses and their aligned lifecycle phases.\n"
                f"- Do not invent new processes or phases. Only return what exists in the knowledge base for the {domain or 'specified'} domain.\n"
                f"- If the capability is not found, politely state: This capability is not defined in the current framework.\n"
                f"- Keep the response concise, structured, and consistent with the examples."
            )

            # Send request
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_text}
            ]

            response = client.chat.completions.create(
                model=config["deployment"],
                temperature=0.2,
                max_tokens=1500,
                top_p=0.9,
                frequency_penalty=0.5,
                messages=messages,
            )

            generated = response.choices[0].message.content.strip()
            def _clean_candidate(text: str) -> str:
                s = text
                s = re.sub(r"```(?:json|yaml)?\n", "", s)
                s = re.sub(r"```", "", s)
                s = s.replace('`', '')
                s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
                s = re.sub(r"\*([^*]+)\*", r"\1", s)
                try:
                    import html
                    s = html.unescape(s)
                except Exception:
                    pass
                s = re.sub(r'""\s*([^"\n\r]+?)\s*""', r'"\1"', s)
                s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)
                # Remove trailing commas in objects/arrays
                s = re.sub(r',\s*(?=[}\]])', '', s)
                # Ensure keys are quoted: foo: -> "foo": (only for simple unquoted keys)
                s = re.sub(r'(?P<prefix>[{,]\s*)(?P<key>[A-Za-z0-9_\- ]+)\s*:(?!\")', lambda m: f"{m.group('prefix')}\"{m.group('key').strip()}\":", s)
                # Normalize smart quotes to normal quotes
                s = s.replace('\u201c', '"').replace('\u201d', '"')
                s = s.replace('\u2018', "'").replace('\u2019', "'")
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
                try:
                    parsed = json.loads(candidate)
                    tried.append('json(cleaned)')
                except Exception:
                    tried.append('json(cleaned) failed')

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
                        candidate_py = candidate.replace('true', 'True').replace('false', 'False').replace('null', 'None')
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

            logger.info(f"Generated ({purpose}) for '{process_name or ''}': parsed keys={list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")

            return {"status": "success", "data": parsed, "raw": generated, "process_name": process_name}

        except Exception as e:
            logger.error(f"Error generating JSON ({purpose}): {str(e)}")
            raise Exception(f"Process generation failed: {str(e)}")


azure_openai_client = AzureOpenAIClient()
openai_client = azure_openai_client
