# Contributing to Pydantic2Django

Thank you for your interest in contributing! This project bridges Python data models and the Django ORM. To keep quality high and avoid regressions, please follow these guidelines.

## Requirements
- Python 3.11+
- UV for dependency management
- Pytest for testing
- Follow PEP 8 and use type hints throughout

## Pull Request Policy
All pull requests must either:
1. Pass the full test suite locally and in CI (GitHub Actions), proving no regressions; or
2. Clearly explain why automated tests are not appropriate for the added/changed behavior, including concrete manual validation steps or alternative verification.

Additionally, PRs should:
- Maintain backward compatibility unless explicitly discussed and approved.
- Keep dependencies minimal and explicit.
- Prefer DRY code and avoid duplicating existing functionality. Search for an existing method before adding a new one.
- Use descriptive names and keep functions single-responsibility.
- Document public APIs with concise docstrings.

## Getting Started (Development)
1. Install uv if needed: https://docs.astral.sh/uv/
2. Install dependencies:
   ```bash
   uv sync
   ```
3. Run tests:
   ```bash
   uv run pytest -q
   ```
4. (Optional) Type-check (mypy is configured):
   ```bash
   uv run mypy
   ```

## PR Checklist (Quick Reference)
- [ ] All tests pass locally: `uv run pytest -q`
- [ ] Tests added/updated for new or changed behavior, or justification provided why tests are not appropriate
- [ ] Backward compatibility preserved (or breaking change is clearly documented and justified)
- [ ] Minimal dependencies; no unnecessary additions
- [ ] Public APIs documented; type hints present
- [ ] Relevant docs updated (e.g., README, guides)

We appreciate your contributions—thank you for helping improve Pydantic2Django!
