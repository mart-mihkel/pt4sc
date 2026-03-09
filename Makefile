REMOTE ?=

pre-commit:
	uv run ruff check --fix
	uv run ruff format
	uv run ty check

upload:
	rsync -rv --exclude-from '.gitignore' . $(REMOTE)

download-log:
	rsync -rv $(REMOTE)/mlflow.db .
