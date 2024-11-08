import kagglehub

# kagglehub.login()

# Replace with path to directory containing model files.
LOCAL_MODEL_DIR = "/home/user/.cache/huggingface/hub/models--dunzhang--stella_en_1.5B_v5/snapshots/221e30586ab5186c4360cbb7aeb643b6efc9d8f8"

MODEL_SLUG = "stella_en_1.5B_v5"  # Replace with model slug.

# Learn more about naming model variations at
# https://www.kaggle.com/docs/models#name-model.
VARIATION_SLUG = "v1"  # Replace with variation slug.

kagglehub.model_upload(
    handle=f"kuto0633/{MODEL_SLUG}/transformers/{VARIATION_SLUG}",
    local_model_dir=LOCAL_MODEL_DIR,
    version_notes="Update 2024-11-08",
)
