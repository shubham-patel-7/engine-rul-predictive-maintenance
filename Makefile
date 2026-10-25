.PHONY: install data train report test app all
install:
	pip install -r requirements.txt && pip install -e .
data:
	python scripts/download_data.py
train:
	python -m rul.train --seeds 0 1 2
report:
	python -m rul.report
test:
	pytest -q
app:
	streamlit run app/streamlit_app.py
all: install data train report test
