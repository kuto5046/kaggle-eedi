# format by ruff
fmt:
    uv run ruff format .

# lint by ruff
lint:
    uv run ruff check --fix .

# type-check by mypy
type-check:
    uv run mypy .

# run pytest in tests directory
test:
    uv run pytest tests

# upload to kaggle dataset
upload:
	uv run python src/tools/upload_code.py
	uv run python src/tools/upload_model.py

upload-model:
	uv run python src/tools/upload_model.py
upload-code:
	uv run python src/tools/upload_code.py

download-model:
	rm /home/user/work/output/kuto-eedi-model.zip
	uv run kaggle datasets download kuto0633/kuto-eedi-model -p /home/user/work/output
	unzip /home/user/work/output/kuto-eedi-model.zip -d /home/user/work/output
# run streamlit app
streamlit:
	uv run streamlit run src/tools/visualizer.py --server.address 0.0.0.0
