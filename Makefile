.PHONY: test e2e lint

test:
	uv run pytest

# Regenerate the committed, byte-reproducible E2E artifact.
e2e:
	uv run python -m apishift.e2e --seed 0 --out artifacts/e2e

lint:
	uv run ruff check apishift tests
