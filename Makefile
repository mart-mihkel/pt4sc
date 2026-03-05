REMOTE ?=

install:
	uv sync
	uv pip install --editable .

pre-commit:
	uv run ruff check --fix
	uv run ruff format
	uv run ty check

marimo:
	uv run marimo edit notebooks

mlflow:
	uv run mlflow server

typst:
	typst watch typesetting/main.typ --open zathura

upload:
	rsync -rv --exclude-from '.gitignore' . $(REMOTE)

download-out:
	rsync -rv $(REMOTE)/out .

download-log:
	rsync -rv $(REMOTE)/mlflow.db .
