# hf_models_free.py
from dotenv import load_dotenv
import os
from huggingface_hub import HfApi

# -------------------------
# Load .env variables
# -------------------------
load_dotenv()
hf_token = os.getenv("HF_API_TOKEN")
if not hf_token:
    raise ValueError("HF_API_TOKEN not found in environment variables")

api = HfApi()

try:
    print("Fetching models from Hugging Face...")
    all_models = api.list_models(token=hf_token)

    # Filter for relevant, free/public LLMs
    relevant_models = []
    for m in all_models:
        tags = m.tags or []
        pipeline_tag = getattr(m, "pipeline_tag", None)
        if pipeline_tag:
            tags.append(pipeline_tag)

        # Keep only text-generation, text2text-generation, conversational
        if any(tag in ["text-generation", "text2text-generation", "conversational"] for tag in tags):
            # Exclude private/restricted models
            if not getattr(m, "private", False):
                relevant_models.append(m.modelId)

    relevant_models = sorted(set(relevant_models))  # remove duplicates

    print("✅ Free/Public LLM models you can use:\n")
    for model in relevant_models:
        print(model)
    print(f"\nTotal free/public relevant models: {len(relevant_models)}")

except Exception as e:
    print(f"❌ Error fetching models: {e}")