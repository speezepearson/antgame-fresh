Always use `uv` to run Python commands.

## PR Validation

All PRs should be validated with:
```bash
uv run mypy . && uv run pytest
```