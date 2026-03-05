# Docs Site (Hugo + Docsy)

This directory contains the Hugo docs app.

Source-of-truth docs remain in `../docs/` (single-source model).
Generated Hugo inputs are written to `site/.generated/` and are not tracked by git.
For internal docs links, use Hugo `relref`/`absURL` helpers; do not hardcode `/docs/...` paths.
Sync generated inputs with:

```bash
.venv/bin/python scripts/docs/sync_hugo_content.py
```

Validate synchronization and links:

```bash
.venv/bin/python scripts/docs/sync_hugo_content.py --check
.venv/bin/python scripts/docs/check_links.py
.venv/bin/python scripts/docs/check_built_output_links.py public
```

Build locally (requires `hugo` and `go`):

```bash
npm install --prefix site
.venv/bin/python scripts/docs/sync_hugo_content.py
hugo --source site --minify --gc --destination ../public
.venv/bin/python scripts/docs/check_built_output_links.py public
```
