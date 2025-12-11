import logging
import google.generativeai as genai
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


def count_tokens(text: str, model: str = "gemini-pro") -> int:
    """Count tokens for Gemini model"""
    try:
        # For Gemini, use the model's built-in token counting
        model_obj = genai.GenerativeModel(model)
        response = model_obj.count_tokens(text)
        return response.total_tokens
    except Exception as e:
        logger.warning(f"Failed to count tokens with Gemini: {e}")
        # Fallback: rough estimation (1 token ≈ 4 characters)
        return len(text) // 4


class GeminiClient:
    def __init__(self):
        self._config = None
        self._client = None

    def _load_config(self):
        """Load config from environment variables"""
        if self._config is None:
            self._config = {
                "api_key": "AIzaSyA6a2MPejOQJYGskVDgST3WbCpE1-V4vVU",
                "model": "gemini-2.5-flash-lite",
            }
            if self._config.get("model"):
                logger.info(f"Loaded Gemini config - Model: {self._config['model']}")
            if self._config.get("api_key"):
                logger.info(f"API Key starts with: {self._config['api_key'][:5]}...")
        return self._config

    def _get_client(self):
        if self._client is None:
            config = self._load_config()
            if not config.get("api_key"):
                raise ValueError(
                    "Missing required Gemini config: api_key. "
                    "Provide this via environment variable GEMINI_API_KEY."
                )
            genai.configure(api_key=config["api_key"])
            self._client = config
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
        """Generate processes for a capability in a specific domain with a given process type using Gemini LLM"""
        # Use the unified generator with a tight, JSON-only system prompt
        schema_example = {
            "capability_name": capability_name,
            "domain": domain,
            "process_type": process_type,
            "processes": [
                {"name": "Example Process", "description": "", "subprocesses": [{"name": "Example Sub", "description": ""}]}
            ]
        }
        prompt_text = f"For the capability '{capability_name}' in the {domain} domain, generate a list of {process_type}-level processes with their subprocesses. Return the result as a JSON object with the following schema (no markdown, no surrounding text):\n{json.dumps(schema_example, indent=2)}"
        return await self.generate_json(prompt_text=prompt_text, purpose="processes", capability_name=capability_name, domain=domain, process_type=process_type)

    async def generate_json(self, *, prompt_text: str, purpose: str = "general", context_sections: Optional[List[str]] = None, capability_name: Optional[str] = None, domain: Optional[str] = None, process_type: Optional[str] = None) -> Dict[str, Any]:
        """Unified generator that requests strict JSON and parses robustly using Gemini.

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

            # Store vaultName from settings if provided (not used by Gemini client
            # currently, but keep for consistency and future secret retrieval).
            vault_url = settings.get("vaultName")
            if vault_url:
                # attach to instance so other methods could use it later
                setattr(self, "key_vault_url", vault_url)

            config = self._load_config()
            self._get_client()

            # Use configured values from settings, fallback to defaults
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

            # Create the model instance
            model = genai.GenerativeModel(
                model_name=config["model"],
                system_instruction=system_prompt,
                generation_config={
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    "top_p": top_p,
                    "top_k": 40,
                }
            )

            # Generate content using Gemini
            response = model.generate_content(prompt_text)
            generated = response.text.strip()

            logger.debug(f"Raw LLM response (first 2000 chars):\n{generated[:2000]}")

            def _clean_candidate(text: str) -> str:
                s = text or ""

                # Remove common fenced code markers and backticks
                s = re.sub(r"```(?:json|yaml)?\n", "", s)
                s = re.sub(r"```", "", s)
                s = s.replace('`', '')

                # Unwrap bold/italic markers
                s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
                s = re.sub(r"\*([^*]+)\*", r"\1", s)

                # Unescape HTML entities
                try:
                    import html
                    s = html.unescape(s)
                except Exception:
                    pass

                # Remove control characters that may break JSON parsing
                s = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", s)

                # Try to extract the first balanced JSON object/array from the text.
                def _extract_first_json(sn: str) -> str:
                    start_idx = None
                    for i, ch in enumerate(sn):
                        if ch in '{[':
                            start_idx = i
                            break
                    if start_idx is None:
                        return sn

                    stack = []
                    in_string = False
                    escape = False
                    for j in range(start_idx, len(sn)):
                        c = sn[j]
                        if escape:
                            escape = False
                            continue
                        if c == '\\':
                            escape = True
                            continue
                        if c == '"':
                            in_string = not in_string
                            continue
                        if in_string:
                            continue
                        if c == '{' or c == '[':
                            stack.append(c)
                        elif c == '}' or c == ']':
                            if not stack:
                                # unmatched closing, give up
                                return sn[start_idx:j+1]
                            opening = stack.pop()
                            if (opening == '{' and c != '}') or (opening == '[' and c != ']'):
                                # mismatched, continue searching
                                continue
                            if not stack:
                                return sn[start_idx:j+1]
                    # If we get here, we didn't find a balanced end; return the substring from start
                    return sn[start_idx:]

                extracted = _extract_first_json(s)

                # Trim and return extracted candidate; do not aggressively rewrite quotes/escapes
                return extracted.strip()

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

            logger.info(f"Generated ({purpose}) for capability '{capability_name or ''}' in domain '{domain or ''}': parsed keys={list(parsed.keys()) if isinstance(parsed, dict) else type(parsed)}")

            return {"status": "success", "data": parsed, "raw": generated, "capability_name": capability_name}

        except Exception as e:
            logger.error(f"Error generating JSON ({purpose}): {str(e)}")
            raise Exception(f"Process generation failed: {str(e)}")


gemini_client = GeminiClient()
