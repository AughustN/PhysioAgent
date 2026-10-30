from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

try:
    from dotenv import dotenv_values
except ImportError:  # pragma: no cover - requirements should provide python-dotenv
    dotenv_values = None


class LLMClientError(RuntimeError):
    pass


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PACKAGE_ROOT.parent
DEFAULT_ENV_FILE = PACKAGE_ROOT / ".env"


@dataclass
class LLMClient:
    mode: str
    base_url: str | None
    model_name: str | None
    api_key: str | None = None
    api_key_env: str | None = None
    provider: str | None = None
    timeout: float = 60.0
    reasoning_effort: str | None = None
    extra_body: dict[str, Any] = field(default_factory=dict)
    last_headers: dict[str, str] = field(default_factory=dict)
    last_usage: dict[str, Any] = field(default_factory=dict)
    """Seconds to wait for a reply.

    60 suits the interactive web path, where a slow answer is worse than no answer. Batch
    callers should raise it: a cold free-tier model behind a gateway can take well over a
    minute to serve its first request, and at 60 s that surfaces as a read timeout the
    caller cannot tell apart from a real outage.
    """

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> "LLMClient":
        llm = settings.get("llm", {})
        env_path = _resolve_env_file(llm.get("env_file"))
        env_values = _load_env_values(env_path)
        settings_mode = str(llm.get("mode", "") or "")
        if _is_offline(settings_mode):
            mode = settings_mode
        else:
            mode = str(_first_value(env_values, "LLM_MODE", os.getenv("LLM_MODE"),
                                    llm.get("mode", "api")))
        provider = _first_value(env_values, "LLM_PROVIDER", os.getenv("LLM_PROVIDER"), llm.get("provider"))
        provider_settings = _provider_settings(llm, provider, strict=_requires_api_key(mode))
        api_key_env = provider_settings.get("api_key_env", llm.get("api_key_env", "OPENAI_API_KEY"))
        api_key = env_values.get(api_key_env)
        if _requires_api_key(mode) and not api_key:
            raise LLMClientError(f"LLM API key is not configured. Set {api_key_env} in .env file: {env_path}")

        return cls(
            mode=mode,
            provider=provider,
            base_url=_first_value(
                env_values,
                "OPENAI_BASE_URL",
                os.getenv("OPENAI_BASE_URL"),
                provider_settings.get("base_url"),
                llm.get("base_url"),
            ),
            model_name=_first_value(
                env_values,
                "MODEL_NAME",
                os.getenv("MODEL_NAME"),
                provider_settings.get("model_name"),
                llm.get("model_name"),
            ),
            api_key=api_key,
            api_key_env=api_key_env,
            extra_body=dict(provider_settings.get("extra_body") or {}),
            reasoning_effort=_first_value(
                env_values,
                "REASONING_EFFORT",
                os.getenv("REASONING_EFFORT"),
                provider_settings.get("reasoning_effort"),
                llm.get("reasoning_effort"),
            ),
        )

    def chat_json(self, system_prompt: str, user_prompt: str,
                  meta: Optional[dict[str, Any]] = None,
                  json_mode: bool = True,
                  schema: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        # json_mode=False drops response_format, for gateways where no route accepts it
        # (cheaperinference zdr deepseek-v4.1-flash, 2026-10-11: 503 "No compatible
        # route"). The reply is then only parsed, not enforced, so callers must validate.
        # schema sends a json_schema response_format instead: vLLM's guided decoding then
        # enforces the keys, their order and any enum, which json_object does not.
        if not self.base_url or not self.model_name:
            raise LLMClientError("LLM base_url/model_name are not configured.")
        if _requires_api_key(self.mode) and not self.api_key:
            env_name = self.api_key_env or "configured API key environment variable"
            raise LLMClientError(f"LLM API key is not configured. Set {env_name}.")

        endpoint = _chat_completions_endpoint(self.base_url)
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        if schema is not None:
            payload["response_format"] = {"type": "json_schema", "json_schema": {
                "name": "reply", "schema": schema, "strict": True}}
        elif json_mode:
            payload["response_format"] = {"type": "json_object"}
        if self.reasoning_effort:
            payload["reasoning_effort"] = self.reasoning_effort
        payload.update(self.extra_body)
        headers = _request_headers(self.api_key)

        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                self.last_headers = dict(response.headers.items())
                body = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = _read_http_error_body(exc)
            detail = f"HTTP {exc.code}: {exc.reason}"
            if body:
                detail += f" - {body}"
            raise LLMClientError(f"LLM API call failed: {detail}") from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMClientError(f"LLM API call failed: {exc}") from exc

        self.last_usage = body.get("usage") or {}
        if meta is not None:
            meta["usage"] = body.get("usage") or {}
            meta["headers"] = self.last_headers
            meta["model"] = body.get("model")
            meta["system_fingerprint"] = body.get("system_fingerprint")
        content = body["choices"][0]["message"]["content"]
        if not json_mode:
            content = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", content or "")
        if meta is not None:
            meta["content"] = content
            # Keys in the order the model wrote them: reasoning written before the dose is
            # part of the explanation claim, and a parsed dict alone cannot show it.
            pairs = json.loads(content, object_pairs_hook=list)
            meta["key_order"] = [p[0] for p in pairs] if isinstance(pairs, list) else []
        return json.loads(content)

    def score_continuations(self, system_prompt: str, user_prompt: str, prefix: str,
                            continuations: list[str]) -> list[float]:
        """Log-probability of each continuation once the reply has reached `prefix`, as the
        served model scores it (no guided decoding). vLLM only: /tokenize renders the chat
        template, and prompt_logprobs scores tokens the request supplies, so a continuation
        of several tokens ("120}") is scored whole."""
        base = (self.base_url or "").rstrip("/")
        root = re.sub(r"/v1$", "", base)
        messages = [{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}]
        head = self._post_json(root + "/tokenize", {"model": self.model_name,
                                                    "messages": messages,
                                                    "add_generation_prompt": True})["tokens"]
        head = head + self._tokens(root, prefix)
        tails = [self._tokens(root, text) for text in continuations]
        body = self._post_json(base + "/completions", {
            "model": self.model_name, "prompt": [head + ids for ids in tails],
            "max_tokens": 1, "temperature": 0, "prompt_logprobs": 1})
        choices = sorted(body["choices"], key=lambda c: c["index"])
        scores = []
        for choice, ids in zip(choices, tails):
            positions = choice["prompt_logprobs"][len(head):]
            scores.append(sum(float(pos[str(token)]["logprob"])
                              for pos, token in zip(positions, ids)))
        return scores

    def _tokens(self, root: str, text: str) -> list[int]:
        return self._post_json(root + "/tokenize", {"model": self.model_name, "prompt": text,
                                                    "add_special_tokens": False})["tokens"]

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"),
                                         headers=_request_headers(self.api_key), method="POST")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = f"HTTP {exc.code}: {exc.reason}"
            body = _read_http_error_body(exc)
            raise LLMClientError(f"LLM API call failed: {detail}"
                                 + (f" - {body}" if body else "")) from exc
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            raise LLMClientError(f"LLM API call failed: {exc}") from exc


def _chat_completions_endpoint(base_url: str) -> str:
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/chat/completions"):
        return endpoint
    return endpoint + "/chat/completions"


def _request_headers(api_key: str | None) -> dict[str, str]:
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "User-Agent": "PhysioAgent/0.1",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


OFFLINE_MODES = {"offline", "mock", "template", "disabled", "none"}


def _is_offline(mode: str) -> bool:
    return str(mode).strip().lower() in OFFLINE_MODES


def _requires_api_key(mode: str) -> bool:
    return str(mode).lower() == "api"


def _provider_settings(llm: dict[str, Any], provider: Any, strict: bool = True) -> dict[str, Any]:
    if not provider:
        return {}
    providers = llm.get("providers") or {}
    provider_name = str(provider)
    if provider_name not in providers:
        if not strict:
            return {}
        raise LLMClientError(f"LLM provider '{provider_name}' is not configured in settings.yaml.")
    provider_settings = providers[provider_name] or {}
    if not isinstance(provider_settings, dict):
        raise LLMClientError(f"LLM provider '{provider_name}' must be a mapping in settings.yaml.")
    return provider_settings


def _read_http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        return exc.read().decode("utf-8", errors="replace")[:500]
    except Exception:
        return ""


def _resolve_env_file(path: str | Path | None) -> Path:
    if path:
        env_path = Path(path)
        if env_path.is_absolute():
            return env_path
        for base in [Path.cwd(), WORKSPACE_ROOT, PACKAGE_ROOT]:
            candidate = (base / env_path).resolve()
            if candidate.exists():
                return candidate
        return (Path.cwd() / env_path).resolve()
    return DEFAULT_ENV_FILE


def _load_env_values(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    if dotenv_values is not None:
        raw = dotenv_values(path)
        return {str(key): str(value) for key, value in raw.items() if key and value is not None}
    return _load_simple_env(path)


def _load_simple_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            values[key] = value
    return values


def _first_value(env_values: dict[str, str], key: str, *fallbacks: Any) -> Any:
    value = env_values.get(key)
    if value not in {None, ""}:
        return value
    for fallback in fallbacks:
        if fallback not in {None, ""}:
            return fallback
    return None
