from __future__ import annotations

from pathlib import Path

from conftest import load_script_module


def test_effective_diversity_script_delegates_to_cli(monkeypatch) -> None:
    module = load_script_module(
        "effective_diversity_audit_script",
        "scripts/effective_diversity_audit.py",
    )
    captured: dict[str, object] = {}

    def _stub_cli_main(argv: list[str]) -> int:
        captured["argv"] = argv
        return 7

    monkeypatch.setattr(module, "cli_main", _stub_cli_main)

    code = module.main(
        [
            "--baseline-config",
            "configs/default.yaml",
            "--variant-config",
            "configs/preset_shift_benchmark_smoke.yaml",
            "--out-dir",
            "tmp/out",
        ]
    )

    assert code == 7
    assert captured["argv"] == [
        "diversity-audit",
        "--baseline-config",
        "configs/default.yaml",
        "--variant-config",
        "configs/preset_shift_benchmark_smoke.yaml",
        "--out-dir",
        "tmp/out",
    ]


def test_sync_docs_helpers_handle_route_aliases_and_heading_attrs() -> None:
    module = load_script_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")

    stripped = module._strip_matching_h1(
        "# How dagzoo Works {#how-dagzoo-works}\n\nBody\n",
        title="How dagzoo Works",
    )
    route_map = module._build_route_map("/dagzoo")

    assert stripped == "Body\n"
    assert route_map["how-it-works.md"] == "/dagzoo/docs/how-it-works/"
    assert route_map["transforms.md"] == "/dagzoo/docs/transforms/"
    assert route_map["how-it-works.html"] == "/dagzoo/docs/how-it-works/"
    assert route_map["transforms.html"] == "/dagzoo/docs/transforms/"


def test_sync_docs_front_matter_includes_aliases_and_page_flags() -> None:
    module = load_script_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")
    meta = module.PAGE_METADATA["how-it-works.md"]

    front_matter = module._front_matter("How It Works", meta)

    assert "title: How It Works" in front_matter
    assert "aliases:" in front_matter
    assert "/canonical/how-it-works.html" in front_matter
    assert "mermaid: true" in front_matter


def test_sync_docs_removes_legacy_generated_paths(tmp_path) -> None:
    module = load_script_module("sync_hugo_content", "scripts/docs/sync_hugo_content.py")
    legacy_dir = tmp_path / "static" / "canonical"
    legacy_dir.mkdir(parents=True)

    changed: list[Path] = []
    module._remove_stale_path(legacy_dir, check=False, changed=changed)

    assert changed == [legacy_dir]
    assert not legacy_dir.exists()


def test_check_repo_paths_passes_for_existing_repo_paths(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    module = load_script_module("check_repo_paths", "scripts/docs/check_repo_paths.py")
    readme = tmp_path / "README.md"
    (tmp_path / "src" / "dagzoo").mkdir(parents=True)
    (tmp_path / "src" / "dagzoo" / "__init__.py").write_text("", encoding="utf-8")
    readme.write_text(
        "Use `src/dagzoo/__init__.py` or `scripts/tool.py [--check]`, and allow `configs/*.yaml`.\n",
        encoding="utf-8",
    )
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "tool.py").write_text("", encoding="utf-8")
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    code = module.main(["README.md"])

    assert code == 0
    assert "Repo path check passed." in capsys.readouterr().out


def test_check_repo_paths_rejects_missing_and_legacy_paths(
    tmp_path,
    monkeypatch,
    capsys,
) -> None:
    module = load_script_module("check_repo_paths", "scripts/docs/check_repo_paths.py")
    readme = tmp_path / "README.md"
    readme.write_text(
        "Legacy `src/dagzoo/cli.py` and missing `src/dagzoo/does_not_exist.py`.\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(module, "REPO_ROOT", tmp_path)

    code = module.main(["README.md"])
    output = capsys.readouterr().out

    assert code == 1
    assert "src/dagzoo/cli.py (legacy path)" in output
    assert "src/dagzoo/does_not_exist.py (missing path)" in output
