"""Shared helpers for title-based Linear backlog seeders."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from github_to_linear import LinearClient, LinearLabel, LinearState, MigrationError


@dataclass(frozen=True, slots=True)
class TicketSpec:
    title: str
    state_name: str
    labels: tuple[str, ...]
    description: str
    parent_title: str | None = None


def required_label_specs(
    ticket_specs: Sequence[TicketSpec],
    *,
    label_colors: Mapping[str, str],
) -> dict[str, str]:
    names = sorted({label for spec in ticket_specs for label in spec.labels})
    missing = [name for name in names if name not in label_colors]
    if missing:
        raise MigrationError(f"Missing configured colors for labels: {', '.join(missing)}")
    return {name: str(label_colors[name]) for name in names}


def find_existing_project_issues(
    linear: LinearClient, project_id: str
) -> dict[str, dict[str, Any]]:
    data = linear.graphql(
        """
        query ProjectIssues($projectId: ID!) {
          issues(filter: { project: { id: { eq: $projectId } } }, first: 200) {
            nodes {
              id
              identifier
              url
              title
            }
          }
        }
        """,
        {"projectId": project_id},
    )
    return {node["title"]: node for node in data["issues"]["nodes"]}


def ensure_labels(
    linear: LinearClient,
    *,
    team_id: str,
    existing_labels: dict[str, LinearLabel],
    label_specs: Mapping[str, str],
) -> dict[str, LinearLabel]:
    labels = dict(existing_labels)
    for name, color in label_specs.items():
        if name in labels:
            continue
        if linear.dry_run:
            print(f"[dry-run] Would create label {name!r}")
            labels[name] = LinearLabel(id=f"dry-run-{name}", name=name, color=color)
            continue
        data = linear.graphql(
            """
            mutation CreateLabel($input: IssueLabelCreateInput!) {
              issueLabelCreate(input: $input) {
                success
                issueLabel {
                  id
                  name
                  color
                }
              }
            }
            """,
            {"input": {"teamId": team_id, "name": name, "color": color}},
        )
        payload = data["issueLabelCreate"]
        if not payload.get("success"):
            raise MigrationError(f"Failed to create label {name!r}")
        node = payload["issueLabel"]
        labels[name] = LinearLabel(id=node["id"], name=node["name"], color=node.get("color"))
    return labels


def create_or_update_issue(
    linear: LinearClient,
    *,
    existing: dict[str, Any] | None,
    title: str,
    description: str,
    team_id: str,
    project_id: str,
    state: LinearState,
    label_ids: list[str],
    parent_id: str | None,
) -> dict[str, str]:
    create_payload: dict[str, Any] = {
        "title": title,
        "description": description,
        "teamId": team_id,
        "projectId": project_id,
        "stateId": state.id,
        "labelIds": label_ids,
    }
    if parent_id:
        create_payload["parentId"] = parent_id

    if existing is None:
        if linear.dry_run:
            print(f"[dry-run] Would create issue {title!r}")
            return {
                "id": f"dry-run-{title}",
                "identifier": "DRY-1",
                "url": "https://linear.app/fake",
            }
        data = linear.graphql(
            """
            mutation CreateIssue($input: IssueCreateInput!) {
              issueCreate(input: $input) {
                success
                issue {
                  id
                  identifier
                  url
                }
                }
            }
            """,
            {"input": create_payload},
        )
        response = data["issueCreate"]
        if not response.get("success"):
            raise MigrationError(f"Failed to create issue {title!r}")
        return response["issue"]

    update_payload: dict[str, Any] = {
        "title": title,
        "description": description,
        "projectId": project_id,
    }
    if parent_id:
        update_payload["parentId"] = parent_id

    if linear.dry_run:
        print(f"[dry-run] Would update issue {title!r} ({existing['identifier']})")
        return {"id": existing["id"], "identifier": existing["identifier"], "url": existing["url"]}
    data = linear.graphql(
        """
        mutation UpdateIssue($id: String!, $input: IssueUpdateInput!) {
          issueUpdate(id: $id, input: $input) {
            success
            issue {
              id
              identifier
              url
            }
            }
        }
        """,
        {"id": existing["id"], "input": update_payload},
    )
    response = data["issueUpdate"]
    if not response.get("success"):
        raise MigrationError(f"Failed to update issue {title!r}")
    return response["issue"]


def seed_ticket_specs(
    linear: LinearClient,
    *,
    ticket_specs: Sequence[TicketSpec],
    states: Mapping[str, LinearState],
    labels: Mapping[str, LinearLabel],
    existing_issues: Mapping[str, dict[str, Any]],
    team_id: str,
    project_id: str,
) -> dict[str, dict[str, str]]:
    created: dict[str, dict[str, str]] = {}
    for spec in ticket_specs:
        state = states.get(spec.state_name)
        if state is None:
            raise MigrationError(f"Missing Linear state {spec.state_name!r}")
        parent_id = None
        if spec.parent_title:
            parent = created.get(spec.parent_title) or existing_issues.get(spec.parent_title)
            if not parent:
                raise MigrationError(f"Parent issue missing: {spec.parent_title}")
            parent_id = parent["id"]
        label_ids = [labels[name].id for name in spec.labels]
        existing = existing_issues.get(spec.title)
        issue = create_or_update_issue(
            linear,
            existing=existing,
            title=spec.title,
            description=spec.description,
            team_id=team_id,
            project_id=project_id,
            state=state,
            label_ids=label_ids,
            parent_id=parent_id,
        )
        created[spec.title] = issue
    return created
