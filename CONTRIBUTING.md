# Contributing

## Local Setup

Install the project and development dependencies:

```bash
python -m pip install -e .[dev]
```

## Fast Verification

Run the fixture-backed smoke and contract suite:

```bash
make smoke
```

Run the full local test suite:

```bash
pytest -q
```

Run lint:

```bash
ruff check src tests scripts
```

## Backend Workflow

The backend is config-driven through [`configs/project.yaml`](./configs/project.yaml).
Main policy lanes and exploratory bidirectional lanes both read from the typed
config loader and shared analysis registry.

Standard backend run:

```bash
make full
```

Key individual steps:

```bash
make download
make build
make qa
make baseline
make dml
make overlap
make app
make report
make public-data
```

## FBI Crime Data

- `make download` includes the normal county-fallback construction path.
- `make build` requires county-level FBI crime by default.
- `python scripts/build_panel.py --allow-missing-county-crime` is only for partial or debugging builds.
- `make fbi-county-fallback` is available when you want to build the county fallback explicitly.

## Public Data And Site

The public site reads [`docs/assets/data/site_data.json`](./docs/assets/data/site_data.json)
directly.

Refresh the checked-in public snapshot after backend changes:

```bash
make public-data
```

Serve the site locally:

```bash
cd docs
python3 -m http.server
```

## Test Expectations

- Keep new workflow tests offline and fixture-backed.
- Prefer extending the existing smoke path instead of adding heavy networked integration tests.
- If you change artifact-producing code, update the shared report-contract tests as part of the same change.
