.PHONY: test lint init-dirs download build qa baseline robustness dml overlap report app reverse reverse-run bidirectional border min-wage-id crime-validation support-trim falsification

test:
	pytest -q

lint:
	ruff check src tests scripts

init-dirs:
	python scripts/download_data.py --init-only

download:
	python scripts/download_data.py

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
