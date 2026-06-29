# AGENTS.md

## Project

MirKatAI — LangGraph-based ReAct agent for miRNA research. Deployed to Vertex AI Agent Engine.

## Package manager

- Use **`uv`** (not pip). Lockfile `uv.lock` committed.
- Install all deps: `uv sync --dev --extra streamlit --extra jupyter --frozen`
- Add a dep: `uv add <package>`
- Python >=3.10,<3.13. Build backend: hatchling.

## Commands

| Action | Command |
|---|---|
| Install | `make install` |
| Playground (Streamlit) | `make playground` |
| All tests | `make test` (runs `tests/unit` then `tests/integration`) |
| Single test | `uv run pytest tests/unit/test_foo.py -v` |
| Lint (order matters) | `uv run codespell && uv run ruff check . --diff && uv run ruff format . --check --diff && uv run mypy .` |
| Deploy to Vertex AI | `make backend` (exports deps, runs `agent_engine_app.py`) |
| Setup dev infra | `PROJECT_ID=<id> make setup-dev-env` |

## Architecture

- **StateGraph** with 6 nodes defined in `app/agent.py` (graph) and `app/nodes.py` (node instances).
- Nodes: `chatbot_router` (orchestrator), `sql_processor_node`, `literature_search_node`, `plot_node`, `execute_tools`, `human_node`.
- Node classes under `app/mirkat/` extend the `node` base class in `node_constructor.py`.
- Instructions live in `app/mirkat/instructions.yaml`, loaded via the `Instructions` enum.

### Routing

The router LLM outputs keyword-prefixed instructions:

- `***ROUTE_TO_SQL***` — database queries
- `***ROUTE_TO_LITERATURE***` — Google-grounded lit search
- `***PLOT***` — generate SVG/PNG plots
- `***ANSWER_DIRECTLY***` — respond from context
- `***FINISH***` — end conversation

## SQL Node quirks

- Manual function calling (`automatic_function_calling=disable=True`), max 10 iterations.
- Large results (>50 rows or >5000 chars) are truncated; model is instructed to use subqueries instead of literal value lists.
- DB env vars: `MIRKAT_USER`, `MIRKAT_PASSWORD`, `MIRKAT_HOST`, `MIRKAT_DATABASE`.

## Environment

- `.env` file at repo root: `GOOGLE_API_KEY` + MySQL credentials.
- `app/langgraph.json` points to `.env` for LangGraph CLI.

## Testing

- Integration tests (`tests/integration/`) require live DB + Gemini API — will fail without valid credentials.
- Plot node unit test is **skipped**: `@pytest.mark.skip("Plot node needs to be attended")`.
- Load tests use Locust in a separate `locust_env` venv (see `tests/load_test/README.md`).

## Logging

- Writes to `mirkat.log` at repo root (UTF-8). Check this first for runtime issues.

## CI/CD

- Cloud Build: `deployment/ci/pr_checks.yaml` runs unit → integration tests sequentially.
- Terraform infra at `deployment/terraform/`.
- CD pipelines in `deployment/cd/` (staging → prod with manual approval).

## Gotchas

- `uv sync --frozen` uses lockfile without updating it. Use plain `uv sync` to update lock.
- `.requirements.txt` is **generated** by `make backend` via `uv export` — not committed.
- Ruff config: line-length 88, ignores E501/C901, enables isort, bugbear, pyupgrade, flake8-comprehensions.
- mypy is strict: disallows untyped defs/calls, no implicit optional, disable_error_code = `misc`, `no-untyped-call`, `no-any-return`.
