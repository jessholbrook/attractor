"""Unified LLM Client: provider-agnostic interface for language models."""
from __future__ import annotations

# Types - Enums
from unified_llm.types.enums import ContentKind, FinishReason, Role, StreamEventType, ToolChoiceMode

# Types - Content
from unified_llm.types.content import (
    AudioData,
    CacheControl,
    ContentPart,
    DocumentData,
    ImageData,
    ThinkingData,
    ToolCallData,
    ToolResultData,
)

# Types - Messages
from unified_llm.types.messages import Message

# Types - Tools
from unified_llm.types.tools import Tool, ToolCall, ToolChoice, ToolResult

# Types - Request/Response
from unified_llm.types.request import Request, ResponseFormat
from unified_llm.types.response import FinishReasonInfo, RateLimitInfo, Response, Usage, Warning

# Types - Streaming
from unified_llm.types.streaming import StreamAccumulator, StreamEvent

# Types - Results
from unified_llm.types.results import GenerateResult, StepResult

# Types - Config
from unified_llm.types.config import (
    AbortController,
    AbortSignal,
    AdapterTimeout,
    RetryPolicy,
    TimeoutConfig,
)

# Errors
from unified_llm.errors import (
    SDKError,
    ProviderError,
    AuthenticationError,
    AccessDeniedError,
    NotFoundError,
    InvalidRequestError,
    RateLimitError,
    ServerError,
    ContentFilterError,
    ContextLengthError,
    QuotaExceededError,
    RequestTimeoutError,
    AbortError,
    NetworkError,
    StreamError,
    InvalidToolCallError,
    NoObjectGeneratedError,
    ConfigurationError,
)

# Core
from unified_llm.adapter import BaseAdapter, ProviderAdapter, StubAdapter
from unified_llm.client import Client, set_default_client, get_default_client

# Middleware
from unified_llm.middleware import CostTracker, logging_middleware, cost_tracking_middleware

# High-Level API
from unified_llm.generate import generate, generate_object
from unified_llm.stream import stream, stream_object

# Catalog
from unified_llm.catalog import get_latest_model, get_model_info, list_models, ModelInfo

# Providers (for direct use)
from unified_llm.providers.openai import OpenAIAdapter
from unified_llm.providers.anthropic import AnthropicAdapter
from unified_llm.providers.gemini import GeminiAdapter

__all__ = [
    # Enums
    "ContentKind",
    "FinishReason",
    "Role",
    "StreamEventType",
    "ToolChoiceMode",
    # Content
    "AudioData",
    "CacheControl",
    "ContentPart",
    "DocumentData",
    "ImageData",
    "ThinkingData",
    "ToolCallData",
    "ToolResultData",
    # Messages
    "Message",
    # Tools
    "Tool",
    "ToolCall",
    "ToolChoice",
    "ToolResult",
    # Request/Response
    "Request",
    "ResponseFormat",
    "FinishReasonInfo",
    "RateLimitInfo",
    "Response",
    "Usage",
    "Warning",
    # Streaming
    "StreamAccumulator",
    "StreamEvent",
    # Results
    "GenerateResult",
    "StepResult",
    # Config
    "AbortController",
    "AbortSignal",
    "AdapterTimeout",
    "RetryPolicy",
    "TimeoutConfig",
    # Errors
    "SDKError",
    "ProviderError",
    "AuthenticationError",
    "AccessDeniedError",
    "NotFoundError",
    "InvalidRequestError",
    "RateLimitError",
    "ServerError",
    "ContentFilterError",
    "ContextLengthError",
    "QuotaExceededError",
    "RequestTimeoutError",
    "AbortError",
    "NetworkError",
    "StreamError",
    "InvalidToolCallError",
    "NoObjectGeneratedError",
    "ConfigurationError",
    # Core
    "BaseAdapter",
    "ProviderAdapter",
    "StubAdapter",
    "Client",
    "set_default_client",
    "get_default_client",
    # Middleware
    "CostTracker",
    "logging_middleware",
    "cost_tracking_middleware",
    # High-Level API
    "generate",
    "generate_object",
    "stream",
    "stream_object",
    # Catalog
    "get_latest_model",
    "get_model_info",
    "list_models",
    "ModelInfo",
    # Providers
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GeminiAdapter",
]
