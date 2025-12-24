Always use `uv` to run Python commands.

## Code Style

- Don't proactively add default values. If adding a default value would clean up many usages, great, add it; but don't add them speculatively.
- Fail fast and loud. If something unexpected happens, don't fall back to some sensible default. Raise an exception.

## Testing

### Test behaviors, not methods.
Bad: `def test_process_transaction(): ...call the function, make many unrelated assertions...`
Good: `class TestProcessTransaction: def test_displays_notification_on_success(): ...; def test_sends_email_when_balance_is_low(): ...`

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