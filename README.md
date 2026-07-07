# Hybrid Multi-Agent Framework Web App (Phase 1)

This project implements **Phase 1: Foundation & Single-Agent State** of the hybrid stateful graph multi-agent framework. It uses **FastAPI**, **Pydantic** for state models, **LangGraph** for cycle orchestration and state persistence, and **LiteLLM** to route LLM completions with elegant local fallback capability.

## Features
- **Centralized Stateful Router (Supervisor)**: Decomposes human intents into structured sub-tasks and delegates execution to specialized worker nodes.
- **Robust State Schemas**: Strongly-typed structures using Pydantic representing user session, task states, and graph state context.
- **State Checkpointing**: Conversational turns and working memories are automatically stored using LangGraph memory checkpointers, allowing full resumption of tasks under thread scopes.
- **Universal LLM & Resilient Fallback**: Integrates LiteLLM for routing calls to Anthropic Claude 3.5, OpenAI GPT-4o, or Gemini models, with a deterministic mock fallback if API keys are missing or invalid (e.g. out of credit).

## Tech Stack
- **FastAPI**: Clean, async REST API layer.
- **Pydantic v2**: Strict data serialization and validation schemas.
- **LangGraph**: Orchestration of cyclical stateful logic.
- **LiteLLM**: High-level unified LLM client interface.

## Directory Structure
- `main.py`: REST endpoints (`POST /chat`, `GET /state/{thread_id}`, `POST /reset/{thread_id}`).
- `state.py`: Strong typing for `Task`, `WorkingMemory`, and `GraphState`.
- `agent.py`: LangGraph StateGraph modeling the cyclical Supervisor-to-Worker communication pattern.
- `llm.py`: Completion router and robust mock fallback.
- `test_app.py`: Comprehensive automated integration and persistence tests.
- `requirements.txt`: Python dependencies.

## Installation & Setup

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Configure environment (Optional)**:
   Add your API keys to a `.env` file:
   ```env
   OPENAI_API_KEY=your-openai-api-key
   GEMINI_API_KEY=your-gemini-api-key
   ```
   *Note: If no API keys are provided or active, the framework gracefully runs on its mock-LLM rendering engine, making it fully testable immediately.*

3. **Run the FastAPI server**:
   ```bash
   uvicorn main:app --reload
   ```

4. **Run the integration test suite**:
   ```bash
   python test_app.py
   ```
