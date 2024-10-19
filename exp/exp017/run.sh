# uv run python -m exp.exp017.search_llm_model llm.model.model=hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4 llm.model.quantization=awq llm.model.dtype=half
# uv run python -m exp.exp017.search_llm_model llm.model.model=microsoft/Phi-3.5-mini-instruct  # oom
# uv run python -m exp.exp017.search_llm_model llm.model.model=microsoft/Phi-3.5-MoE-instruct llm.model.quantization=awq  # oom
uv run python -m exp.exp017.search_llm_model llm.model.model=google/gemma-2-2b-it llm.model.max_model_len=null
# uv run python -m exp.exp017.search_llm_model llm.model.model=mistralai/Mistral-7B-Instruct-v0.1
# uv run python -m exp.exp017.search_llm_model llm.model.model=mistralai/Mixtral-8x7B-Instruct-v0.1 # oom
# uv run python -m exp.exp017.search_llm_model llm.model.model=meta-llama/Llama-3.2-3B-Instruct
