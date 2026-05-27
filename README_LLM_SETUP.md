# LLM planner setup for CO2RR XAS Agent

The local parser can process simple natural-language requests with regex. For flexible language such as:

```text
energy range start from -5 eV to 50 eV with step size 0.2 eV
```

use the LLM planner.

## Install dependency

```bash
pip install openai pydantic
```

## Environment variables

For any OpenAI-compatible endpoint, including institutional gateways that expose the OpenAI chat/completions API:

```bash
export CO2RR_LLM_MODEL="your-model-name"
export CO2RR_LLM_API_KEY="your-token-or-api-key"
export CO2RR_LLM_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

Equivalent fallback names are also accepted:

```bash
export OPENAI_MODEL="your-model-name"
export OPENAI_API_KEY="your-token-or-api-key"
export OPENAI_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

For ALCF-style variables, these are also accepted:

```bash
export ALCF_MODEL="your-model-name"
export ALCF_API_KEY="your-token"
export ALCF_BASE_URL="https://your-openai-compatible-endpoint/v1"
```

If your ALCF/AskSage endpoint does not expose an OpenAI-compatible API, keep `agent/schemas.py` unchanged and replace only the `_chat()` method in `agent/planner.py` with the institution-specific client call.

## Usage

```python
from agent.co2rr_xas_agent import process_request

result = process_request(
    "Generate FDMNES Green multiple scattering XAS inputs without SCF, "
    "with Quadrupole and Spinorbit for CO on Cu(111), radius 7 Angstrom, "
    "energy range start from -5 eV to 50 eV with step size 0.2 eV, "
    "K edge, NERSC account m5268",
    output_dir="generated_outputs/fdmnes_green_test",
    use_llm=True,
)
```

`use_llm=True` first tries the LLM planner. If it cannot initialize or the model call fails, the agent prints a warning and falls back to `LocalIntentParser`.

## Debugging the planner only

```python
from agent.planner import create_llm_planner_from_env

planner = create_llm_planner_from_env()
plan = planner.plan(
    "Generate FDMNES Green multiple scattering XAS inputs without SCF, "
    "with energy range start from -5 eV to 50 eV with step size 0.2 eV"
)
print(plan.model_dump())
print(plan.to_parsed_intent("debug_output"))
```

Expected parsed energy object:

```json
{"emin": -5.0, "emax": 50.0, "step": 0.2}
```

Expected execution parameter for FDMNES:

```python
(-5.0, 0.2, 50.0)
```
