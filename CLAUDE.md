# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Kimi CLI** is a Python AI agent CLI (and optional web UI) built by Moonshot AI. It provides an agentic loop (`KimiSoul`) that accepts user tasks, calls LLMs, executes tools, and streams output back to the terminal or a web UI.

This repository is a **local fork** of `MoonshotAI/kimi-cli` with Azure OpenAI provider support added (`azure_kimi`, `azure_openai_legacy_router`). Keep changes minimal to ease upstream merges.

## Commands

All workflows go through the Makefile:

```bash
# Setup
make prepare              # Install all deps + prek git hooks

# Development
uv run kimi               # Run the agent interactively
uv run kimi --help        # CLI reference
make web-back             # FastAPI backend on :5494
make web-front            # Vite dev server (web/)

# Code quality (also run automatically on commit via prek)
make format               # Auto-format with ruff + biome
make check                # Ruff lint + pyright + ty type checks

# Testing
make test                 # All test suites
make test-kimi-cli        # tests/ + tests_e2e/
make test-kosong          # Kosong package (includes doctests)
make test-pykaos          # PyKAOS package

# Building
make build                # Full build (builds web first, then Python packages)
make build-bin            # PyInstaller one-file standalone binary
```

Run a single pytest test file: `uv run pytest tests/path/to/test_file.py -v`

Type checking is **strict** for `src/kimi_cli/**` via pyright. Line length is 100.

## Architecture

### Layer overview

```
CLI (typer)  →  KimiCLI (app.py)  →  KimiSoul (soul/)  →  Kosong (packages/kosong)  →  LLM providers
                                    ↕
                               KimiToolset (tools/)
                                    ↕
                            Wire protocol (wire/)  →  UI (ui/ or web/)
```

### Key modules

- **`src/kimi_cli/soul/kimisoul.py`** — Core agentic loop. Drives the LLM ↔ tool round-trips, handles context compaction (`SimpleCompaction`), and retry logic (Tenacity).
- **`src/kimi_cli/llm.py`** — Resolves named models/providers from config and constructs `kosong` client instances. All provider logic (failover, first-token timeout, env-var overrides) lives here.
- **`src/kimi_cli/config.py`** — Pydantic models for `~/.kimi/config.toml`. Defines `ProviderConfig`, `ModelConfig`, `AgentConfig`, etc.
- **`src/kimi_cli/tools/`** — Every callable tool (files, shell, web, multiagent, todo, dmail). Tools extend `kosong.tooling.CallableTool2` with Pydantic params.
- **`src/kimi_cli/wire/`** — Transport-agnostic message protocol (JSON-RPC). Transports: file-based (ACP), WebSocket, server. `WireMessage` is the canonical message type.
- **`src/kimi_cli/ui/`** — Terminal UI (Rich), print mode, ACP mode (Zed/JetBrains), wire mode.
- **`packages/kosong/`** — LLM abstraction: unifies message formats, streaming, and tool-calling across providers (Kimi, OpenAI, Anthropic, Gemini, Vertex).
- **`packages/kaos/`** — OS abstraction: local and SSH file/command operations via a unified `KaosPath` interface.
- **`web/`** — React/TypeScript frontend (Vite). Built artifacts are embedded in the Python package.

### Provider / config system

Providers and models are declared in `~/.kimi/config.toml`:

```toml
[[providers]]
name = "my-provider"
type = "kimi"           # kimi | openai_legacy | openai_responses | azure_openai_legacy_router | anthropic | google_genai
base_url = "https://api.moonshot.ai/v1"
api_key_env = "KIMI_API_KEY"   # preferred over hardcoded api_key
first_token_warn_seconds = 5

[[models]]
name = "my-model"
model = "kimi-k2-turbo"
provider = "my-provider"
```

`api_key_env` (recently added) lets you reference an environment variable instead of storing keys in TOML.

The `azure_openai_legacy_router` provider (local fork addition) wraps multiple `OpenAILegacy` providers in a `FailoverChatProvider` for multi-region failover with first-token timeout and warn logging.

### Azure AI Foundry / Kimi-K2.5 on Azure

Kimi-K2.5 is a **"Direct from Azure" partner model** deployed via Azure AI Foundry. Key facts:

- **Endpoints**: Use `.cognitiveservices.azure.com/openai/deployments/<name>` with `api-version` query param. The newer `/openai/v1/` path is **incompatible** with the openai Python SDK (v2.14.0) due to `httpx.URL.join` stripping the path prefix.
- **Auth**: `api-key` header (set automatically by `_openai_client_kwargs` when an Azure base_url is detected).
- **Thinking mode**: Uses `reasoning_key = "reasoning_content"` and `reasoning_effort = "high"`.
- **No PTU available**: Partner models can't use Provisioned Throughput Units. Pay-per-token only, with undocumented rate limits.
- **Intermittent outages**: The Kimi-K2.5 inference backend can go unresponsive across all regions simultaneously. Azure's content filter still responds (HTTP 200 + `prompt_filter_results` SSE event) but no model tokens follow. GPT models on the same resource continue working during these outages.
- **Failover architecture**: `FailoverChatProvider` (`packages/kosong/src/kosong/contrib/chat_provider/failover.py`) wraps the entire generate-to-first-token sequence in a single timeout. Warns periodically (visibility) and hard-timeouts for failover.
- **Logging**: kosong module logging is always enabled (`app.py`) so failover warnings appear in `~/.kimi/logs/kimi.log` even without `--debug`.
- **Retry**: `kimisoul.py`'s `_is_retryable_error` handles `TimeoutError` (from failover) in addition to `APITimeoutError`, `APIConnectionError`, etc.

#### Potential next steps (not yet implemented)

- **Circuit breaker**: Skip recently-failed endpoints instead of retrying them every request. Track consecutive failures per endpoint with a cooldown (e.g., 5 min).
- **Azure Monitor alerts**: Set up latency/error alerts on both AI Services resources.
- **Entra ID auth**: Replace API key auth with `DefaultAzureCredential` for better security (won't fix reliability).
- **`/openai/v1/` migration**: `httpx.URL.join` test showed path-prefix stripping, but one successful end-to-end run completed before Kimi-K2.5 went down. Retest when Azure is stable — the SDK may handle it differently in practice. Code support for v1 detection (`_is_azure_openai_v1_base_url`) is already in `llm.py`.
- **Contact Azure Support**: Ask about Kimi-K2.5 capacity/SLA and whether Global Standard routing improves availability.

### Testing providers

Use `type = "_echo"`, `_scripted_echo`, or `_chaos` in config for deterministic/fault-injection testing without real API keys.
