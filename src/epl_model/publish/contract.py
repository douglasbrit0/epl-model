"""JSON contract validation and all-or-nothing local snapshot publication."""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

from jsonschema import Draft202012Validator, FormatChecker
import requests


FILES = {
    "latest.json": "latest.schema.json",
    "fixtures.json": "fixtures.schema.json",
    "ratings.json": "ratings.schema.json",
    "season.json": "season.schema.json",
    "performance.json": "performance.schema.json",
}


def validate_payloads(payloads: dict[str, Any], schemas_dir: Path) -> None:
    """Validate every contract document before any publication side effect."""
    if set(payloads) != set(FILES):
        raise ValueError(f"Contract payload names must be exactly {sorted(FILES)}")
    for filename, schema_name in FILES.items():
        schema = json.loads((schemas_dir / schema_name).read_text(encoding="utf-8"))
        errors = sorted(Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payloads[filename]), key=lambda error: list(error.path))
        if errors:
            locations = ["/".join(str(part) for part in error.path) or "$" for error in errors[:5]]
            raise ValueError(f"{filename} failed {schema_name} validation at {locations}: {errors[0].message}")


def append_prediction_log(log_path: Path, records: list[dict[str, Any]]) -> None:
    """Append immutable prediction records; duplicate prediction IDs are rejected."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if log_path.exists():
        with log_path.open(encoding="utf-8") as stream:
            for line in stream:
                if line.strip():
                    existing.add(json.loads(line)["prediction_id"])
    new_ids = [record["prediction_id"] for record in records]
    if len(new_ids) != len(set(new_ids)) or existing.intersection(new_ids):
        raise ValueError("Prediction log append contains a duplicate prediction_id")
    with log_path.open("a", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _upload_supabase_snapshot(payloads: dict[str, Any]) -> None:
    """Upload immutable version files, then atomically advance latest.json."""
    import os

    base_url = os.getenv("SUPABASE_URL")
    if not base_url:
        return
    key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    if not key:
        raise RuntimeError("SUPABASE_SERVICE_ROLE_KEY is required when SUPABASE_URL is set")
    bucket = os.getenv("SUPABASE_BUCKET", "epl-model")
    latest = payloads["latest.json"]
    version = latest["generated_at"].replace(":", "-")
    root = f"{base_url.rstrip('/')}/storage/v1/object"
    headers = {"Authorization": f"Bearer {key}", "apikey": key, "Content-Type": "application/json"}
    for filename, payload in payloads.items():
        if filename == "latest.json":
            continue
        path = f"snapshots/{version}/{filename}"
        url = f"{root}/{quote(bucket, safe='')}/{quote(path, safe='/')}"
        response = requests.post(url, headers={**headers, "x-upsert": "false"}, json=payload, timeout=30)
        response.raise_for_status()
    pointer_url = f"{root}/{quote(bucket, safe='')}/latest.json"
    response = requests.post(pointer_url, headers={**headers, "x-upsert": "true"}, json=latest, timeout=30)
    response.raise_for_status()


def publish_snapshot(payloads: dict[str, Any], output_dir: Path, schemas_dir: Path, prediction_records: list[dict[str, Any]]) -> None:
    """Validate to staging, append the audit log, then atomically replace output dir."""
    validate_payloads(payloads, schemas_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent))
    backup = output_dir.with_name(f".{output_dir.name}.previous")
    try:
        for filename, payload in payloads.items():
            (stage / filename).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        append_prediction_log(output_dir.parent / "logs" / "prediction_log.jsonl", prediction_records)
        _upload_supabase_snapshot(payloads)
        if backup.exists():
            shutil.rmtree(backup)
        if output_dir.exists():
            os.replace(output_dir, backup)
        try:
            os.replace(stage, output_dir)
        except Exception:
            if backup.exists() and not output_dir.exists():
                os.replace(backup, output_dir)
            raise
        if backup.exists():
            shutil.rmtree(backup)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
