"""Discoverable Langfuse CLI access; keys stay in the subprocess environment."""
from __future__ import annotations

import json
import os
import subprocess

from ..core.tracing import load_environment, redact


def api(resource, action, *args, body=None):
    load_environment()
    env = dict(os.environ)
    env["LANGFUSE_HOST"] = env.get("LANGFUSE_BASE_URL") or env.get("LANGFUSE_HOST", "")
    command = ["npx.cmd" if os.name == "nt" else "npx", "--yes", "langfuse-cli", "api", resource, action, *args, "--json"]
    if body is not None:
        command += ["--body-file", "-"]
    result = subprocess.run(command, input=json.dumps(body, ensure_ascii=False) if body is not None else None,
                            capture_output=True, text=True, encoding="utf-8", env=env, timeout=90)
    if result.returncode:
        raise RuntimeError(redact(result.stderr.strip()))
    payload = json.loads(result.stdout)
    if payload.get("status", 500) >= 400:
        raise RuntimeError(f"Langfuse {resource}/{action} failed: {redact(payload.get('body'))}")
    return payload["body"]


def all_items(resource, action, *args):
    result = api(resource, action, *args, "--all", "--max-items", "10000")
    # CLI --all returns the collected array, normal list calls use {data, meta}.
    if isinstance(result, list):
        return result
    return result["data"]
