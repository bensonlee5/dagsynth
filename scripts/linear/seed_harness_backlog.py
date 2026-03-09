#!/usr/bin/env python3
"""Seed recurring harness-audit tracker objects in Linear."""

from __future__ import annotations

import argparse
from pathlib import Path

from github_to_linear import (
    DEFAULT_LINEAR_ENDPOINT,
    LinearClient,
    LinearState as _LinearState,
    load_linear_api_key,
)
from seed_linear_utils import (
    TicketSpec,
    create_or_update_issue as _create_or_update_issue,
    ensure_labels as _ensure_labels,
    find_existing_project_issues,
    required_label_specs as _required_label_specs,
    seed_ticket_specs,
)


HARNESS_ARTICLE_URL = "https://openai.com/index/harness-engineering/"
RECURRING_AUDIT_TITLE = "ops(harness): weekly full-repo harness audit"
HARNESS_EPIC_TITLE = "epic: harness engineering adoption for autonomous dagzoo development"
LABEL_COLORS = {
    "audit": "#7C3AED",
    "documentation": "#0EA5E9",
    "epic": "#DC2626",
    "harness": "#2563EB",
}

LinearState = _LinearState
create_or_update_issue = _create_or_update_issue


def _markdown_list(items: list[str]) -> str:
    return "\n".join(f"- {item}" for item in items)


def build_ticket_specs() -> list[TicketSpec]:
    epic_description = (
        "## Summary\n\n"
        "Adopt Harness Engineering best practices for `dagzoo` so autonomous agents can work "
        "against a clearer repo contract, more enforceable architecture, and lower-entropy "
        "tooling/process surfaces.\n\n"
        "Primary reference:\n\n"
        f"- {HARNESS_ARTICLE_URL}\n\n"
        "## Themes\n\n"
        + _markdown_list(
            [
                "repo as system of record",
                "agent legibility",
                "agent-first interfaces",
                "architecture and taste enforcement",
                "entropy and garbage collection",
                "safe increases in autonomy",
            ]
        )
        + "\n\n## Acceptance Criteria\n\n"
        + _markdown_list(
            [
                "Child tickets exist for the concrete harness gaps already visible in the repo.",
                "The recurring weekly audit is configured and points at the repo-owned rubric.",
                "The adoption work is tracked separately from feature roadmap epics.",
            ]
        )
    )

    child_specs = [
        TicketSpec(
            title="docs(harness): expand AGENTS.md into a complete agent operating contract",
            state_name="Backlog",
            labels=("harness", "documentation"),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The current `AGENTS.md` is intentionally short and does not yet provide the level "
                "of repo legibility described in Harness Engineering.\n\n"
                "## Goal\n\n"
                "Turn `AGENTS.md` into a high-signal contract for autonomous contributors.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Canonical bootstrap, test, lint, and release commands are documented.",
                        "Public vs internal surfaces are called out where they matter.",
                        "Issue/PR expectations and user-facing break policy are explicit.",
                        "The document is specific to dagzoo rather than generic agent advice.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="ci(harness): add a pull request template and handoff evidence requirements",
            state_name="Backlog",
            labels=("harness",),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo has review expectations, but no checked-in PR template or standard handoff evidence surface.\n\n"
                "## Goal\n\n"
                "Force consistent intent/risk/validation reporting in PRs.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "A repo PR template exists.",
                        "The template requires change summary, validation evidence, and user-facing break callouts.",
                        "The template aligns with the existing `/review` expectation in `AGENTS.md`.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="ci(architecture): add structural dependency checks around src/dagzoo/core",
            state_name="Backlog",
            labels=("harness",),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo documents a preferred `src/dagzoo/core`-centric dependency direction, but there is no "
                "automated enforcement against drift.\n\n"
                "## Goal\n\n"
                "Add structural checks that keep architecture intent enforceable.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "A repeatable check exists for dependency-direction drift or obvious cycles.",
                        "The enforced rules match the repo's stated architecture conventions.",
                        "Violations fail locally and in CI with actionable output.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="ci(docs): enforce system-of-record docs updates for user-facing changes",
            state_name="Backlog",
            labels=("harness", "documentation"),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo has policy around user-facing breaks, but no explicit guardrail that required docs updates "
                "happen alongside CLI/output-contract changes.\n\n"
                "## Goal\n\n"
                "Add a docs/system-of-record guardrail for user-facing changes.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Changes to CLI flags, persisted metadata schema, or artifact contracts trigger docs expectations.",
                        "Version-bump/changelog policy is checked where applicable.",
                        "The guardrail is narrow enough to avoid excessive false positives.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="tooling(harness): add one-command doctor and verify entrypoints for agents",
            state_name="Backlog",
            labels=("harness",),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "Canonical commands are spread across CI configs, docs, and scripts. There is no minimal doctor/verify "
                "surface an unfamiliar agent can trust immediately.\n\n"
                "## Goal\n\n"
                "Provide a canonical bootstrap/verification entrypoint pair.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "One command verifies environment sanity and required tools.",
                        "One command runs the standard local validation path.",
                        "Docs and CI reference the same canonical surfaces.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="docs(harness): write a repo-wide harness-engineering guide for dagzoo contributors",
            state_name="Backlog",
            labels=("harness", "documentation"),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo has a weekly audit rubric, but it still lacks one broader dagzoo-specific guide that "
                "translates Harness Engineering into contributor-facing standards across docs, process, and tooling.\n\n"
                "## Goal\n\n"
                "Create the repo-owned harness-engineering guide for dagzoo contributors.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "The guide references the Harness Engineering article directly.",
                        "The guide explains dagzoo-specific expectations beyond the weekly audit rubric.",
                        "The guidance connects the repo contract, workflow, and verification surfaces in one place.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
        TicketSpec(
            title="cleanup(harness): eliminate stale generated-output/doc drift and codify garbage collection",
            state_name="Backlog",
            labels=("harness",),
            parent_title=HARNESS_EPIC_TITLE,
            description=(
                "## Current Gap\n\n"
                "The repo already documents at least one stale-output hazard (`public/` versus `site/public`) and does "
                "not yet have a fully explicit garbage-collection policy for stale generated surfaces.\n\n"
                "## Goal\n\n"
                "Reduce entropy by making stale outputs and cleanup rules explicit and enforceable.\n\n"
                "## Acceptance Criteria\n\n"
                + _markdown_list(
                    [
                        "Known stale-output/documentation drift is resolved or clearly fenced.",
                        "Cleanup rules for generated/local artifacts are documented.",
                        "Weekly audit checks can verify this area deterministically.",
                    ]
                )
                + f"\n\n## Reference\n\n- {HARNESS_ARTICLE_URL}\n"
            ),
        ),
    ]

    audit_description = (
        "## Summary\n\n"
        "Run the full weekly harness audit for `dagzoo`, using the repo-owned rubric to detect drift, "
        "create remediation issues, and keep the repo legible for unattended agent work.\n\n"
        "Primary reference:\n\n"
        f"- {HARNESS_ARTICLE_URL}\n"
        "- `docs/development/harness_audit.md`\n"
        "- `docs/development/issue_authoring.md`\n\n"
        "## Schedule\n\n"
        "- Weekly\n"
        "- Friday\n"
        "- 10:00 PM `America/Los_Angeles`\n\n"
        "## Required Flow\n\n"
        "1. Audit the repo using the checklist in `docs/development/harness_audit.md`.\n"
        "2. Reuse existing open issues when the same gap is already tracked.\n"
        "3. Create one remediation issue per net-new actionable gap.\n"
        "4. Put remediation issues in `Backlog` with label `harness`.\n"
        "5. Link remediation issues from this audit ticket.\n"
        "6. Close only after a concise summary is written.\n\n"
        "## Completion Rule\n\n"
        "This ticket is complete only when it records either:\n\n"
        "- linked remediation issues for every net-new actionable gap, or\n"
        "- a short explicit note that no action was required this week.\n"
    )

    return [
        TicketSpec(
            title=HARNESS_EPIC_TITLE,
            state_name="Backlog",
            labels=("epic", "harness"),
            description=epic_description,
        ),
        *child_specs,
        TicketSpec(
            title=RECURRING_AUDIT_TITLE,
            state_name="Todo",
            labels=("audit", "harness"),
            description=audit_description,
        ),
    ]


def required_label_specs() -> dict[str, str]:
    return _required_label_specs(build_ticket_specs(), label_colors=LABEL_COLORS)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--linear-api-key-file", required=True, type=Path)
    parser.add_argument("--project-slug", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_LINEAR_ENDPOINT)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def ensure_labels(
    linear: LinearClient,
    *,
    team_id: str,
    existing_labels: dict[str, object],
) -> dict[str, object]:
    return _ensure_labels(
        linear,
        team_id=team_id,
        existing_labels=existing_labels,
        label_specs=required_label_specs(),
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    api_key = load_linear_api_key(args.linear_api_key_file)
    linear = LinearClient(api_key, endpoint=args.endpoint, dry_run=args.dry_run)
    project_id, team_id, _team_key = linear.get_project(args.project_slug)
    states, labels = linear.get_team_metadata(team_id)
    states = linear.ensure_workflow_states(team_id, states)
    labels = ensure_labels(linear, team_id=team_id, existing_labels=labels)
    existing_issues = find_existing_project_issues(linear, project_id)

    created = seed_ticket_specs(
        linear,
        ticket_specs=build_ticket_specs(),
        states=states,
        labels=labels,
        existing_issues=existing_issues,
        team_id=team_id,
        project_id=project_id,
    )

    audit = created[RECURRING_AUDIT_TITLE]
    print(f"AUDIT_ISSUE_URL={audit['url']}")
    print(
        "NOTE=Configure this issue as a native recurring Linear issue: weekly, Friday, 10:00 PM America/Los_Angeles."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
