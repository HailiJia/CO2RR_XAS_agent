from pathlib import Path

script = r'''#!/usr/bin/env python3
"""
ALCF-powered runner for the CO2RR XAS Python agent.

What it does:
1. Sends your natural-language request to an ALCF inference endpoint.
2. Converts the request into a small JSON execution plan.
3. Calls your local CO2RR XAS backend: process_request(...)
4. Writes generated files into the selected output directory.
5. Saves a run summary JSON.

Expected repo layout:
co2rr-xas-agent/
├── agent/
│   ├── __init__.py
│   └── co2rr_xas_agent.py
├── tools/
│   ├── __init__.py
│   ├── structure_generator.py
│   ├── xas_input_generator.py
│   ├── result_parser.py
│   └── utils.py
├── inference_auth_token.py
└── alcf_co2rr_runner.py
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict

from openai import OpenAI


SYSTEM_PROMPT = """
You are a CO2RR XAS workflow planner.

Convert the user's request into a compact JSON object for a local Python backend.
Return JSON only. Do not use markdown. Do not add explanations.

Allowed JSON fields:
{
  "canonical_request": string,
  "output_dir": string,
  "nersc_account": string or null,
  "nersc_queue": string or null,
  "nersc_nodes": integer or null,
  "nersc_walltime": string or null,
  "email": string or null,
  "cluster_radius": number or null
}

Rules:
- canonical_request should preserve the scientific intent in plain English.
- If the user specifies an output directory, put it in output_dir.
- If output_dir is not specified, use the output_dir provided by the caller.
- If queue is not specified, use "regular".
- If NERSC account is not specified, use null.
- Do not claim files were created. The local Python backend will create files.
"""


def project_dir() -> Path:
    return Path(__file__).resolve().parent


def get_access_token() -> str:
    """
    Get ALCF inference access token using inference_auth_token.py.

    This avoids relying on the helper script's import API; it just calls the
    same command you tested successfully:
        python inference_auth_token.py get_access_token
    """
    helper = project_dir() / "inference_auth_token.py"
    if not helper.exists():
        raise FileNotFoundError(
            f"Cannot find {helper}. Download it first with:\n"
            "curl -L -o inference_auth_token.py "
            "https://raw.githubusercontent.com/argonne-lcf/inference-endpoints/refs/heads/main/inference_auth_token.py"
        )

    token = subprocess.check_output(
        [sys.executable, str(helper), "get_access_token"],
        text=True,
    ).strip()

    if not token:
        raise RuntimeError("inference_auth_token.py returned an empty token.")

    return token


def get_base_url(cluster: str) -> str:
    cluster = cluster.lower()
    if cluster == "sophia":
        return "https://inference-api.alcf.anl.gov/resource_server/sophia/vllm/v1"
    if cluster == "metis":
        return "https://inference-api.alcf.anl.gov/resource_server/metis/api/v1"
    raise ValueError("cluster must be either 'sophia' or 'metis'")


def extract_json(text: str) -> Dict[str, Any]:
    """
    Robustly parse JSON even if the model accidentally returns code fences.
    """
    text = text.strip()

    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.lower().startswith("json"):
            text = text[4:].strip()

    start = text.find("{")
    end = text.rfind("}")

    if start == -1 or end == -1 or end <= start:
        raise ValueError(f"Could not find a JSON object in model output:\n{text}")

    return json.loads(text[start : end + 1])


def make_client(cluster: str) -> OpenAI:
    return OpenAI(
        api_key=get_access_token(),
        base_url=get_base_url(cluster),
    )


def llm_plan(
    user_prompt: str,
    default_output_dir: str,
    cluster: str,
    model: str,
) -> Dict[str, Any]:
    client = make_client(cluster)

    prompt = (
        f"Default output_dir from caller: {default_output_dir}\n\n"
        f"User request:\n{user_prompt}"
    )

    response = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=800,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )

    text = response.choices[0].message.content
    plan = extract_json(text)

    if not plan.get("canonical_request"):
        raise ValueError(f"Model plan missing canonical_request:\n{json.dumps(plan, indent=2)}")

    if not plan.get("output_dir"):
        plan["output_dir"] = default_output_dir

    if not plan.get("nersc_queue"):
        plan["nersc_queue"] = "regular"

    return plan


def import_backend():
    """
    Import your local CO2RR XAS backend.

    Preferred layout:
        from agent.co2rr_xas_agent import process_request

    Fallback layout:
        from co2rr_xas_agent import process_request
    """
    sys.path.insert(0, str(project_dir()))

    try:
        from agent.co2rr_xas_agent import process_request  # type: ignore
        return process_request
    except ImportError as first_error:
        try:
            from co2rr_xas_agent import process_request  # type: ignore
            return process_request
        except ImportError as second_error:
            raise ImportError(
                "Could not import your backend.\n\n"
                "Expected either:\n"
                "  agent/co2rr_xas_agent.py\n"
                "or:\n"
                "  co2rr_xas_agent.py\n\n"
                "Your co2rr_xas_agent.py imports tools.* modules, so the clean layout is:\n"
                "  agent/co2rr_xas_agent.py\n"
                "  tools/structure_generator.py\n"
                "  tools/xas_input_generator.py\n"
                "  tools/result_parser.py\n"
                "  tools/utils.py\n"
            ) from second_error


def run_backend(plan: Dict[str, Any]) -> Dict[str, Any]:
    process_request = import_backend()

    output_dir = plan["output_dir"]

    kwargs = {}
    for key in [
        "nersc_account",
        "nersc_queue",
        "nersc_nodes",
        "nersc_walltime",
        "email",
        "cluster_radius",
    ]:
        value = plan.get(key)
        if value is not None:
            kwargs[key] = value

    return process_request(
        plan["canonical_request"],
        output_dir=output_dir,
        **kwargs,
    )


def save_summary(output_dir: str, summary: Dict[str, Any]) -> Path:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    summary_path = out / "_agent_run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary_path


def maybe_zip_output(output_dir: str) -> str:
    output_path = Path(output_dir).resolve()
    zip_base = str(output_path)
    zip_path = shutil.make_archive(zip_base, "zip", root_dir=output_path)
    return zip_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the CO2RR XAS agent using an ALCF inference model as the planner."
    )
    parser.add_argument(
        "prompt",
        help="Natural-language request, e.g. 'Generate XAS inputs for CO on Cu(111)'",
    )
    parser.add_argument(
        "--output-dir",
        default="generated_outputs/test_run",
        help="Where generated files should be written.",
    )
    parser.add_argument(
        "--cluster",
        default="sophia",
        choices=["sophia", "metis"],
        help="ALCF inference cluster.",
    )
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-20b",
        help=(
            "Model name. For Sophia use e.g. openai/gpt-oss-20b or openai/gpt-oss-120b. "
            "For Metis use e.g. gpt-oss-120b."
        ),
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Create a zip archive of the output directory after the run.",
    )

    args = parser.parse_args()

    print("Planning with ALCF inference endpoint...")
    plan = llm_plan(
        user_prompt=args.prompt,
        default_output_dir=args.output_dir,
        cluster=args.cluster,
        model=args.model,
    )

    print("LLM plan:")
    print(json.dumps(plan, indent=2))

    print("\nRunning local CO2RR XAS backend...")
    backend_result = run_backend(plan)

    summary = {
        "prompt": args.prompt,
        "cluster": args.cluster,
        "model": args.model,
        "llm_plan": plan,
        "backend_result": backend_result,
    }

    summary_path = save_summary(plan["output_dir"], summary)

    print("\nDone.")
    print(f"Output directory: {Path(plan['output_dir']).resolve()}")
    print(f"Run summary: {summary_path.resolve()}")

    if args.zip:
        zip_path = maybe_zip_output(plan["output_dir"])
        print(f"Zip archive: {zip_path}")


if __name__ == "__main__":
    main()
'''

path = Path("/mnt/data/alcf_co2rr_runner.py")
path.write_text(script, encoding="utf-8")
print(f"Created {path}")