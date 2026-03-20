.PHONY: test lint smoke init-dirs download fbi-county-fallback build qa baseline robustness dml overlap report app public-data full reverse reverse-run bidirectional border min-wage-id crime-validation support-trim falsification

test:
	pytest -q

lint:
	ruff check src tests scripts

smoke:
	pytest -q tests/test_config.py tests/test_analysis_registry.py tests/test_fbi_crime.py tests/test_dml.py tests/test_overlap.py tests/test_report_contracts.py tests/test_refresh_public_data.py tests/test_app_artifacts.py tests/test_final_report.py tests/test_smoke_pipeline.py tests/test_bidirectional_runner.py

init-dirs:
	python scripts/download_data.py --init-only

download:
	python scripts/download_data.py

fbi-county-fallback:
	python scripts/build_fbi_county_fallback.py

build:
	python scripts/build_panel.py

qa:
	python scripts/build_qa_report.py

baseline:
	python scripts/run_baseline.py

robustness:
	python scripts/run_robustness.py

dml:
	python scripts/run_dml.py --n-folds 3

overlap:
	python scripts/build_overlap_diagnostics.py

report:
	python scripts/build_final_report.py

app:
	python scripts/build_app_artifacts.py

public-data:
	python scripts/refresh_public_data.py

full:
	python scripts/download_data.py
	python scripts/build_panel.py
	python scripts/build_qa_report.py
	python scripts/run_baseline.py
	python scripts/run_dml.py --n-folds 3
	python scripts/build_overlap_diagnostics.py
	python scripts/build_final_report.py
	python scripts/build_app_artifacts.py
	python scripts/refresh_public_data.py

reverse:
	python scripts/build_reverse_direction_scaffold.py

reverse-run:
	python scripts/run_reverse_direction.py

bidirectional:
	python scripts/run_bidirectional_poverty_crime.py --n-folds 3

border:
	python scripts/run_border_design.py

min-wage-id:
	python scripts/run_min_wage_identification.py

crime-validation:
	python scripts/build_crime_measurement_validation.py

support-trim:
	python scripts/run_support_trimmed_dml.py --n-folds 3

falsification:
	python scripts/run_falsification.py
