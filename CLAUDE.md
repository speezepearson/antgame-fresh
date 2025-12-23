Always use `uv` to run Python commands.

## Package Management

When adding new package dependencies, use:
```bash
uv add <package-name>
```

## PR Validation

All PRs should be validated with:
```bash
uv run mypy . && uv run pytest
```