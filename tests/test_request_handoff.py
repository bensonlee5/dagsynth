from __future__ import annotations

import json

import pytest

from dagzoo.config import REQUEST_FILE_VERSION_V1, REQUEST_TASK_CLASSIFICATION, RequestFileConfig
from dagzoo.core.request_handoff import (
    REQUEST_HANDOFF_SCHEMA_NAME,
    REQUEST_HANDOFF_SCHEMA_VERSION,
    build_request_handoff_manifest,
    validate_request_handoff_manifest,
    write_request_handoff_manifest,
)


def _request_config(output_root: str) -> RequestFileConfig:
    return RequestFileConfig.from_dict(
        {
            "version": REQUEST_FILE_VERSION_V1,
            "task": REQUEST_TASK_CLASSIFICATION,
            "dataset_count": 2,
            "rows": 1024,
            "profile": "default",
            "output_root": output_root,
        }
    )


def test_build_request_handoff_manifest_is_versioned_and_valid(tmp_path) -> None:
    request = _request_config(output_root="requests/demo")
    payload = build_request_handoff_manifest(
        request_path=tmp_path / "request.yaml",
        request=request,
        run_root=tmp_path / "run",
        generated_dir=tmp_path / "run" / "generated",
        filter_dir=tmp_path / "run" / "filter",
        filtered_corpus_dir=tmp_path / "run" / "curated",
        effective_config_path=tmp_path / "run" / "generated" / "effective_config.yaml",
        effective_config_trace_path=tmp_path / "run" / "generated" / "effective_config_trace.yaml",
        filter_manifest_path=tmp_path / "run" / "filter" / "filter_manifest.ndjson",
        filter_summary_path=tmp_path / "run" / "filter" / "filter_summary.json",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        filter_total_datasets=2,
        filter_accepted_datasets=1,
        filter_rejected_datasets=1,
        filter_elapsed_seconds=6.0,
        filter_datasets_per_minute=20.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
    )

    validate_request_handoff_manifest(payload)

    assert payload["schema_name"] == REQUEST_HANDOFF_SCHEMA_NAME
    assert payload["schema_version"] == REQUEST_HANDOFF_SCHEMA_VERSION
    assert payload["request"]["payload"] == request.to_dict()
    assert payload["artifacts"]["filtered_corpus_dir"] == str(
        (tmp_path / "run" / "curated").resolve()
    )
    assert payload["summary"]["acceptance_rate"] == pytest.approx(0.5)
    assert payload["throughput"]["generation_stage"]["datasets_per_minute"] == pytest.approx(10.0)
    assert payload["diversity_artifacts"] == {
        "summary_json_path": None,
        "summary_md_path": None,
    }


def test_write_request_handoff_manifest_writes_json_and_rejects_invalid_payload(tmp_path) -> None:
    request = _request_config(output_root="requests/demo")
    manifest_path = write_request_handoff_manifest(
        request_path=tmp_path / "request.yaml",
        request=request,
        run_root=tmp_path / "run",
        generated_dir=tmp_path / "run" / "generated",
        filter_dir=tmp_path / "run" / "filter",
        filtered_corpus_dir=tmp_path / "run" / "curated",
        effective_config_path=tmp_path / "run" / "generated" / "effective_config.yaml",
        effective_config_trace_path=tmp_path / "run" / "generated" / "effective_config_trace.yaml",
        filter_manifest_path=tmp_path / "run" / "filter" / "filter_manifest.ndjson",
        filter_summary_path=tmp_path / "run" / "filter" / "filter_summary.json",
        generated_datasets=2,
        generation_elapsed_seconds=12.0,
        filter_total_datasets=2,
        filter_accepted_datasets=2,
        filter_rejected_datasets=0,
        filter_elapsed_seconds=4.0,
        filter_datasets_per_minute=30.0,
        requested_device="cpu",
        resolved_device="cpu",
        hardware_backend="cpu",
        hardware_device_name="CPU",
        hardware_tier="cpu",
        hardware_policy="none",
        out_path=tmp_path / "run" / "handoff_manifest.json",
    )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    validate_request_handoff_manifest(payload)
    assert payload["summary"]["accepted_datasets"] == 2

    payload["request"]["payload"]["version"] = "v2"
    with pytest.raises(
        ValueError, match=r"handoff_manifest.request.payload: version must be one of"
    ):
        validate_request_handoff_manifest(payload)
