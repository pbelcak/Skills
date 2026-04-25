import os
import threading
from typing import Any, Literal

from openai import AsyncOpenAI, OpenAI
from dotenv import load_dotenv

load_dotenv()


class LMQueryError(Exception):
    """Raised when an LM endpoint query fails."""
    pass


ReasoningEffort = Literal["none", "minimal", "low", "medium", "high", "xhigh"]
Verbosity = Literal["low", "medium", "high"]


class LM:
    """Wrapper around an OpenAI-compatible chat completions endpoint, bound to a specific model."""

    def __init__(self, model: str, api_key: str | None = None, base_url: str = "https://inference-api.nvidia.com", max_tokens: int | None = None, temperature: float | None = None, **kwargs):
        """Create a client for ``model``. Defaults to NVAPI_KEY and NVIDIA's inference API.
        If ``max_tokens`` is set, it is used as the default token output limit for all queries.
        If ``temperature`` is set, it is used as the default sampling temperature (e.g. 0.0–2.0).
        kwargs are forwarded to the underlying API call.
        """
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        self.usage: dict[str, dict[str, int]] = {}
        self._usage_lock = threading.Lock()
        api_key = api_key or os.environ["NVAPI_KEY"]
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.async_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    def _record_usage(self, response) -> None:
        """Record token usage from a response into this instance's usage dict."""
        usage = getattr(response, "usage", None)
        if usage is None:
            return
        inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
        out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
        if inp or out:
            with self._usage_lock:
                if self.model not in self.usage:
                    self.usage[self.model] = {
                        "input_tokens": 0, "output_tokens": 0,
                        "cache_reused_input_tokens": 0, "cache_new_input_tokens": 0,
                    }
                self.usage[self.model]["input_tokens"] += inp
                self.usage[self.model]["output_tokens"] += out

    def _defaults(self, **kwargs) -> dict:
        """Merge instance defaults into kwargs (caller's kwargs take precedence)."""
        if self.max_tokens is not None and "max_tokens" not in kwargs:
            kwargs = {**kwargs, "max_tokens": self.max_tokens}
        if self.temperature is not None and "temperature" not in kwargs:
            kwargs = {**kwargs, "temperature": self.temperature}
        if self.kwargs is not None:
            kwargs = {**kwargs, **self.kwargs}
        return kwargs

    def query(self, prompt: str, **kwargs) -> str:
        """Send a single user prompt and return the response text."""
        kwargs = self._defaults(**kwargs)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            self._record_usage(response)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery(self, prompt: str, **kwargs) -> str:
        """Async version of ``query``."""
        kwargs = self._defaults(**kwargs)
        try:
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                **kwargs,
            )
            self._record_usage(response)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    def query_messages(self, messages: list[dict], **kwargs) -> str:
        """Send a full message list and return the response text."""
        kwargs = self._defaults(**kwargs)
        try:
            response = self.client.chat.completions.create( # type: ignore
            model=self.model,
            messages=messages, # type: ignore
            **kwargs,
            )
            self._record_usage(response)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery_messages(self, messages: list[dict], **kwargs) -> str:
        """Async version of ``query_messages``."""
        kwargs = self._defaults(**kwargs)
        try:
            response = await self.async_client.chat.completions.create( # type: ignore
            model=self.model,
            messages=messages, # type: ignore
            **kwargs,
        )
            self._record_usage(response)
            return response.choices[0].message.content or ""
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    def query_messages_raw(self, messages: list[dict], **kwargs):
        """Sync query returning the raw ``ChatCompletion`` response object."""
        kwargs = self._defaults(**kwargs)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore
                **kwargs,
            )
            self._record_usage(response)
            return response
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery_messages_raw(self, messages: list[dict], **kwargs):
        """Async query returning the raw ``ChatCompletion`` response object."""
        kwargs = self._defaults(**kwargs)
        try:
            response = await self.async_client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore
            **kwargs,
            )
            self._record_usage(response)
            return response
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e


def _messages_to_responses_input(messages: list[dict]) -> tuple[list[dict], str | None]:
    """Convert chat completions-style messages to Responses API input list and instructions.

    Handles:
    - system messages -> instructions string
    - user/assistant messages -> message items
    - assistant messages with tool_calls -> function_call items
    - tool role messages -> function_call_output items
    """
    import json as _json

    input_items: list[dict] = []
    instructions: str | None = None
    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")
        if isinstance(content, list):
            text = next(
                (c.get("text", str(c)) for c in content if isinstance(c, dict)),
                str(content),
            )
        else:
            text = content

        if role == "system":
            if instructions is None:
                instructions = text
            else:
                instructions = instructions + "\n" + text
        elif role == "tool":
            # Tool result -> function_call_output item
            tool_call_id = m.get("tool_call_id", "")
            input_items.append({
                "type": "function_call_output",
                "call_id": tool_call_id,
                "output": text,
            })
        elif role == "assistant" and m.get("tool_calls"):
            # Assistant message with tool calls
            # First add any text content as a message
            if text:
                input_items.append({"type": "message", "role": "assistant", "content": text})
            # Then add each tool call as a function_call item
            for tc in m["tool_calls"]:
                fn = tc.get("function", {})
                args = fn.get("arguments", "{}")
                if isinstance(args, str):
                    try:
                        args = _json.loads(args)
                    except (ValueError, TypeError):
                        args = {}
                input_items.append({
                    "type": "function_call",
                    "call_id": tc.get("id", ""),
                    "name": fn.get("name", ""),
                    "arguments": args,
                })
        else:
            input_items.append({"type": "message", "role": role, "content": text})
    return input_items, instructions


def _response_output_text(response) -> str:
    """Extract assistant text from a Responses API response including any errors and reasoning."""
    if getattr(response, "error", None) is not None:
        raise RuntimeError(f"OpenAI Responses API error: {response.error}")
    ret = ""
    for item in response.output:
        if item.type == "reasoning":
            ret += "Reasoning: " + item.content if item.content else "Empty." + "\n"
        if item.type == "message" and item.content:
            for part in item.content:
                if part.type == "output_text":
                    return part.text
        
    return ""


class OpenAILM(LM):
    """LM implementation using the OpenAI Responses API (client.responses) with configurable reasoning effort and verbosity."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://inference-api.nvidia.com",
        max_tokens: int | None = None,
        reasoning_effort: ReasoningEffort | None = None,
        verbosity: Verbosity | None = None,
        **kwargs,
    ):
        """Create a client for ``model`` using the OpenAI Responses API.
        Uses NVIDIA Inference API key by default. ``reasoning_effort`` and ``verbosity``
        are applied to every request unless overridden in kwargs.
        """
        api_key = api_key or os.environ.get("NVAPI_KEY")
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            **kwargs,
        )
        self.reasoning_effort = reasoning_effort
        self.verbosity = verbosity

    def _defaults(self, **kwargs) -> dict:
        kwargs = super()._defaults(**kwargs)
        if "max_tokens" in kwargs:
            kwargs["max_output_tokens"] = kwargs.pop("max_tokens")
        reasoning_effort = kwargs.pop("reasoning_effort", self.reasoning_effort)
        verbosity = kwargs.pop("verbosity", self.verbosity)
        if reasoning_effort is not None and "reasoning" not in kwargs:
            kwargs["reasoning"] = {"effort": reasoning_effort, "summary": "detailed"}
        if verbosity is not None:
            kwargs["text"] = {**(kwargs.get("text") or {}), "verbosity": verbosity}
        return kwargs

    def query(self, prompt: str, **kwargs) -> str:
        kwargs = self._defaults(**kwargs)
        try:
            response = self.client.responses.create(
            model=self.model,
            input=prompt,
            **kwargs,
        )
            self._record_usage(response)
            return _response_output_text(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery(self, prompt: str, **kwargs) -> str:
        kwargs = self._defaults(**kwargs)
        try:
            response = await self.async_client.responses.create(
            model=self.model,
            input=prompt,
            **kwargs,
            )
            self._record_usage(response)
            return _response_output_text(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    def query_messages(self, messages: list[dict], **kwargs) -> str:
        input_items, instructions = _messages_to_responses_input(messages)
        try:
            kwargs = self._defaults(**kwargs)
            create_kwargs: dict = {k: v for k, v in kwargs.items() if k not in ("messages",)}
            if instructions is not None:
                create_kwargs["instructions"] = instructions
            response = self.client.responses.create(
                model=self.model,
                input=input_items,
                **create_kwargs,
            )
            self._record_usage(response)
            return _response_output_text(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery_messages(self, messages: list[dict], **kwargs) -> str:
        input_items, instructions = _messages_to_responses_input(messages)
        kwargs = self._defaults(**kwargs)
        create_kwargs = {k: v for k, v in kwargs.items() if k not in ("messages",)}
        if instructions is not None:
            create_kwargs["instructions"] = instructions
        try:
            response = await self.async_client.responses.create(
            model=self.model,
            input=input_items,
            **create_kwargs,
        )
            self._record_usage(response)
            return _response_output_text(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    def query_messages_raw(self, messages: list[dict], **kwargs):
        """Sync query returning the raw Responses API response object."""
        input_items, instructions = _messages_to_responses_input(messages)
        kwargs = self._defaults(**kwargs)
        create_kwargs = {k: v for k, v in kwargs.items() if k not in ("messages",)}
        if instructions is not None:
            create_kwargs["instructions"] = instructions
        try:
            response = self.client.responses.create(
                model=self.model,
                input=input_items,
                **create_kwargs,
            )
            self._record_usage(response)
            return response
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery_messages_raw(self, messages: list[dict], **kwargs):
        """Async query returning the raw Responses API response object."""
        input_items, instructions = _messages_to_responses_input(messages)
        kwargs = self._defaults(**kwargs)
        create_kwargs = {k: v for k, v in kwargs.items() if k not in ("messages",)}
        if instructions is not None:
            create_kwargs["instructions"] = instructions
        try:
            response = await self.async_client.responses.create(
            model=self.model,
            input=input_items,
            **create_kwargs,
        )
            self._record_usage(response)
            return response
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e


# ---------------------------------------------------------------------------
# Anthropic-native LM (uses anthropic SDK via NVIDIA proxy)
# ---------------------------------------------------------------------------


def _openai_tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert OpenAI-format tool schemas to Anthropic format."""
    out = []
    for t in tools:
        if t.get("type") == "function":
            fn = t["function"]
            out.append({
                "name": fn["name"],
                "description": fn.get("description", ""),
                "input_schema": fn.get("parameters", {"type": "object", "properties": {}}),
            })
        else:
            out.append(t)
    return out


def _openai_messages_to_anthropic(messages: list[dict]) -> tuple[str | None, list[dict]]:
    """Split system message out and convert the rest to Anthropic format.

    Returns (system_prompt, anthropic_messages).
    """
    system_parts: list[str] = []
    anthropic_msgs: list[dict] = []

    for m in messages:
        role = m.get("role", "user")
        content = m.get("content", "")

        if role == "system":
            system_parts.append(content if isinstance(content, str) else str(content))
            continue

        if role == "tool":
            # OpenAI tool results -> Anthropic tool_result content block in a user message
            tool_call_id = m.get("tool_call_id", "")
            block = {
                "type": "tool_result",
                "tool_use_id": tool_call_id,
                "content": content if isinstance(content, str) else str(content),
            }
            # Merge into the last user message if it exists, otherwise create one
            if anthropic_msgs and anthropic_msgs[-1]["role"] == "user":
                last_content = anthropic_msgs[-1]["content"]
                if isinstance(last_content, list):
                    last_content.append(block)
                else:
                    anthropic_msgs[-1]["content"] = [{"type": "text", "text": last_content}, block]
            else:
                anthropic_msgs.append({"role": "user", "content": [block]})
            continue

        if role == "assistant" and "tool_calls" in m and m["tool_calls"]:
            # Convert OpenAI assistant tool_calls to Anthropic content blocks
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in m["tool_calls"]:
                fn = tc["function"]
                import json as _json
                try:
                    tool_input = _json.loads(fn["arguments"])
                except (ValueError, TypeError):
                    tool_input = {}
                blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": fn["name"],
                    "input": tool_input,
                })
            anthropic_msgs.append({"role": "assistant", "content": blocks})
            continue

        # Regular user/assistant message
        anthropic_msgs.append({"role": role, "content": content})

    system = "\n".join(system_parts) if system_parts else None
    return system, anthropic_msgs


class _FakeToolCall:
    """Mimics OpenAI's tool call object for compatibility with Cache."""

    def __init__(self, tc_id: str, name: str, arguments: str):
        self.id = tc_id
        self.function = type("Function", (), {"name": name, "arguments": arguments})()


class _FakeMessage:
    """Mimics OpenAI's ChatCompletionMessage for compatibility with Cache."""

    def __init__(self, content: str | None, tool_calls: list[_FakeToolCall] | None):
        self.content = content
        self.tool_calls = tool_calls if tool_calls else None


class _FakeChoice:
    """Mimics OpenAI's Choice for compatibility with Cache."""

    def __init__(self, message: _FakeMessage, finish_reason: str):
        self.message = message
        self.finish_reason = finish_reason


class _FakeChatCompletion:
    """Mimics OpenAI's ChatCompletion so Cache's ``hasattr(response, 'choices')`` path works."""

    def __init__(self, choice: _FakeChoice, thinking: str | None = None):
        self.choices = [choice]
        self.thinking = thinking


def _anthropic_response_to_fake_completion(response) -> _FakeChatCompletion:
    """Convert an Anthropic Messages response to a fake OpenAI ChatCompletion."""
    import json as _json

    text_parts: list[str] = []
    tool_calls: list[_FakeToolCall] = []
    thinking_text: str | None = None

    for block in response.content:
        if block.type == "thinking":
            thinking_text = block.thinking
        elif block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append(_FakeToolCall(
                tc_id=block.id,
                name=block.name,
                arguments=_json.dumps(block.input),
            ))

    content = "\n".join(text_parts) if text_parts else None

    if response.stop_reason == "tool_use":
        finish_reason = "tool_calls"
    elif response.stop_reason == "end_turn":
        finish_reason = "stop"
    elif response.stop_reason == "max_tokens":
        finish_reason = "length"
    else:
        finish_reason = response.stop_reason or "stop"

    message = _FakeMessage(content=content, tool_calls=tool_calls or None)
    choice = _FakeChoice(message=message, finish_reason=finish_reason)
    return _FakeChatCompletion(choice=choice, thinking=thinking_text)


class AnthropicLM(LM):
    """LM using the Anthropic Messages API (via NVIDIA proxy) with extended thinking support."""

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        base_url: str = "https://inference-api.nvidia.com",
        max_tokens: int | None = None,
        temperature: float | None = None,
        thinking_budget: int | None = None,
        **kwargs,
    ):
        """Create a client using the Anthropic SDK pointed at the NVIDIA proxy.

        ``thinking_budget``: if set, enables extended thinking with this many
        budget tokens. Thinking content is returned via the ``thinking``
        attribute on the fake ChatCompletion returned by ``*_raw`` methods.
        Set to ``None`` (default) to disable thinking.
        """
        api_key = api_key or os.environ["NVAPI_KEY"]
        super().__init__(
            model,
            api_key=api_key,
            base_url=base_url,
            max_tokens=max_tokens,
            temperature=temperature,
            **kwargs,
        )
        import anthropic
        self.thinking_budget = thinking_budget
        self.anthropic_client = anthropic.Anthropic(api_key=api_key, base_url=base_url)
        self.anthropic_async_client = anthropic.AsyncAnthropic(api_key=api_key, base_url=base_url)

    def _anthropic_kwargs(self, **kwargs) -> dict:
        """Build kwargs for an Anthropic messages.create call."""
        kw: dict[str, Any] = {"model": self.model}
        kw["max_tokens"] = kwargs.pop("max_tokens", self.max_tokens) or 4096

        temperature = kwargs.pop("temperature", self.temperature)
        thinking_budget = kwargs.pop("thinking_budget", self.thinking_budget)
        if thinking_budget is not None:
            kw["thinking"] = {"type": "enabled", "budget_tokens": thinking_budget}
            kw["temperature"] = 1.0  # Anthropic requires temperature=1 with thinking
        elif temperature is not None:
            kw["temperature"] = temperature

        return kw

    def _extract_text(self, response) -> str:
        """Extract the text content from an Anthropic response."""
        for block in response.content:
            if block.type == "text":
                return block.text
        return ""

    def query(self, prompt: str, **kwargs) -> str:
        kw = self._anthropic_kwargs(**kwargs)
        try:
            response = self.anthropic_client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                **kw,
            )
            self._record_usage(response)
            return self._extract_text(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery(self, prompt: str, **kwargs) -> str:
        kw = self._anthropic_kwargs(**kwargs)
        try:
            response = await self.anthropic_async_client.messages.create(
                messages=[{"role": "user", "content": prompt}],
                **kw,
            )
            self._record_usage(response)
            return self._extract_text(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    def query_messages(self, messages: list[dict], **kwargs) -> str:
        kw = self._anthropic_kwargs(**kwargs)
        system, anthropic_msgs = _openai_messages_to_anthropic(messages)
        if system is not None:
            kw["system"] = system
        try:
            response = self.anthropic_client.messages.create(
                messages=anthropic_msgs,
                **kw,
            )
            self._record_usage(response)
            return self._extract_text(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery_messages(self, messages: list[dict], **kwargs) -> str:
        kw = self._anthropic_kwargs(**kwargs)
        system, anthropic_msgs = _openai_messages_to_anthropic(messages)
        if system is not None:
            kw["system"] = system
        try:
            response = await self.anthropic_async_client.messages.create(
                messages=anthropic_msgs,
                **kw,
            )
            self._record_usage(response)
            return self._extract_text(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    def query_messages_raw(self, messages: list[dict], **kwargs):
        """Returns a fake ChatCompletion wrapping the Anthropic response.

        Compatible with Cache's ``hasattr(response, 'choices')`` code path.
        Thinking summary (if enabled) is available as ``response.thinking``.
        """
        tools = kwargs.pop("tools", None)
        kw = self._anthropic_kwargs(**kwargs)
        system, anthropic_msgs = _openai_messages_to_anthropic(messages)
        if system is not None:
            kw["system"] = system
        if tools:
            kw["tools"] = _openai_tools_to_anthropic(tools)
        try:
            response = self.anthropic_client.messages.create(
                messages=anthropic_msgs,
                **kw,
            )
            self._record_usage(response)
            return _anthropic_response_to_fake_completion(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e

    async def aquery_messages_raw(self, messages: list[dict], **kwargs):
        """Async version of query_messages_raw."""
        tools = kwargs.pop("tools", None)
        kw = self._anthropic_kwargs(**kwargs)
        system, anthropic_msgs = _openai_messages_to_anthropic(messages)
        if system is not None:
            kw["system"] = system
        if tools:
            kw["tools"] = _openai_tools_to_anthropic(tools)
        try:
            response = await self.anthropic_async_client.messages.create(
                messages=anthropic_msgs,
                **kw,
            )
            self._record_usage(response)
            return _anthropic_response_to_fake_completion(response)
        except Exception as e:
            raise LMQueryError(f"Error querying {self.model}: {e}") from e