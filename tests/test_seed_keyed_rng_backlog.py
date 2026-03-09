from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_module(name: str, relative_path: str):
    path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load_module("github_to_linear", "scripts/linear/github_to_linear.py")
_load_module("seed_linear_utils", "scripts/linear/seed_linear_utils.py")
MODULE = _load_module("seed_keyed_rng_backlog", "scripts/linear/seed_keyed_rng_backlog.py")


class _RecordingLinear:
    def __init__(
        self, *, dry_run: bool = False, existing_issues: list[dict[str, str]] | None = None
    ) -> None:
        self.dry_run = dry_run
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.existing_issues = list(existing_issues or [])

    def get_project(self, project_slug: str) -> tuple[str, str, str]:
        _ = project_slug
        return ("project-1", "team-1", "DAG")

    def get_team_metadata(self, team_id: str) -> tuple[dict[str, object], dict[str, object]]:
        _ = team_id
        return (
            {
                "Backlog": MODULE.LinearState(id="state-backlog", name="Backlog", type="backlog"),
                "Todo": MODULE.LinearState(id="state-todo", name="Todo", type="unstarted"),
            },
            {},
        )

    def ensure_workflow_states(
        self,
        team_id: str,
        states: dict[str, object],
    ) -> dict[str, object]:
        _ = team_id
        return dict(states)

    def graphql(self, query: str, variables: dict[str, object]) -> dict[str, object]:
        self.calls.append((query, dict(variables)))
        if "query ProjectIssues" in query:
            return {"issues": {"nodes": list(self.existing_issues)}}
        if "issueLabelCreate" in query:
            input_payload = variables["input"]
            assert isinstance(input_payload, dict)
            label_name = str(input_payload["name"])
            return {
                "issueLabelCreate": {
                    "success": True,
                    "issueLabel": {
                        "id": f"label-{label_name}",
                        "name": label_name,
                        "color": input_payload.get("color"),
                    },
                }
            }
        if "issueCreate" in query:
            input_payload = variables["input"]
            assert isinstance(input_payload, dict)
            title = str(input_payload["title"])
            slug = title.split(":", maxsplit=1)[0].replace("(", "-").replace(")", "")
            return {
                "issueCreate": {
                    "success": True,
                    "issue": {
                        "id": f"issue-{slug}",
                        "identifier": f"DAG-{len(self.calls)}",
                        "url": f"https://linear.app/dagzoo/issue/{slug}",
                    },
                }
            }
        if "issueUpdate" in query:
            return {
                "issueUpdate": {
                    "success": True,
                    "issue": {
                        "id": str(variables["id"]),
                        "identifier": "DAG-EXISTING",
                        "url": "https://linear.app/dagzoo/issue/existing",
                    },
                }
            }
        raise AssertionError(f"Unexpected GraphQL query: {query}")


def test_build_ticket_specs_match_keyed_rng_plan() -> None:
    specs = MODULE.build_ticket_specs()
    assert [spec.title for spec in specs] == [
        "design(rng): define keyed RNG contract and migration plan for semantic reproducibility",
        "refactor(rng): add keyed RNG primitives and hierarchical namespace helpers",
        "refactor(execution): move typed plans and executor paths onto keyed RNG",
        "refactor(runtime): move orchestration, noise, missingness, and postprocess onto keyed RNG",
        "hardening(repro): prove keyed semantic reproducibility and update docs",
    ]
    assert specs[0].state_name == "Todo"
    assert all(spec.parent_title == MODULE.KEYED_RNG_EPIC_TITLE for spec in specs)
    assert specs[-1].labels == ("rng", "documentation")


def test_required_label_specs_cover_rng_and_documentation() -> None:
    assert MODULE.required_label_specs() == {
        "documentation": "#0EA5E9",
        "rng": "#2563EB",
    }


def test_main_requires_existing_parent_epic(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    linear = _RecordingLinear(dry_run=True, existing_issues=[])
    monkeypatch.setattr(MODULE, "load_linear_api_key", lambda _path: "test-key")
    monkeypatch.setattr(MODULE, "LinearClient", lambda *args, **kwargs: linear)

    with pytest.raises(MODULE.MigrationError, match=MODULE.KEYED_RNG_EPIC_TITLE):
        MODULE.main(
            [
                "--linear-api-key-file",
                str(tmp_path / "linear.key"),
                "--project-slug",
                "proj",
                "--dry-run",
            ]
        )


def test_main_rerun_preserves_existing_state_and_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing_issues = [
        {
            "id": "epic-1",
            "identifier": "DAG-EPIC",
            "url": "https://linear.app/dagzoo/issue/epic",
            "title": MODULE.KEYED_RNG_EPIC_TITLE,
        },
        {
            "id": "issue-existing",
            "identifier": "DAG-101",
            "url": "https://linear.app/dagzoo/issue/dag-101",
            "title": "refactor(runtime): move orchestration, noise, missingness, and postprocess onto keyed RNG",
        },
    ]
    linear = _RecordingLinear(dry_run=False, existing_issues=existing_issues)
    monkeypatch.setattr(MODULE, "load_linear_api_key", lambda _path: "test-key")
    monkeypatch.setattr(MODULE, "LinearClient", lambda *args, **kwargs: linear)

    exit_code = MODULE.main(
        [
            "--linear-api-key-file",
            str(tmp_path / "linear.key"),
            "--project-slug",
            "proj",
        ]
    )

    assert exit_code == 0
    update_calls = [call for call in linear.calls if "issueUpdate" in call[0]]
    assert len(update_calls) == 1
    _query, variables = update_calls[0]
    input_payload = variables["input"]
    assert isinstance(input_payload, dict)
    assert input_payload == {
        "title": "refactor(runtime): move orchestration, noise, missingness, and postprocess onto keyed RNG",
        "description": next(
            spec.description
            for spec in MODULE.build_ticket_specs()
            if spec.title
            == "refactor(runtime): move orchestration, noise, missingness, and postprocess onto keyed RNG"
        ),
        "projectId": "project-1",
        "parentId": "epic-1",
    }


def test_main_dry_run_prints_seeded_issue_lines(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    linear = _RecordingLinear(
        dry_run=True,
        existing_issues=[
            {
                "id": "epic-1",
                "identifier": "DAG-EPIC",
                "url": "https://linear.app/dagzoo/issue/epic",
                "title": MODULE.KEYED_RNG_EPIC_TITLE,
            }
        ],
    )
    monkeypatch.setattr(MODULE, "load_linear_api_key", lambda _path: "test-key")
    monkeypatch.setattr(MODULE, "LinearClient", lambda *args, **kwargs: linear)

    exit_code = MODULE.main(
        [
            "--linear-api-key-file",
            str(tmp_path / "linear.key"),
            "--project-slug",
            "proj",
            "--dry-run",
        ]
    )

    captured = capsys.readouterr()
    assert exit_code == 0
    assert captured.out.count("SEEDED_ISSUE=DRY-1 https://linear.app/fake") == 5
