import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Load model and tokenizer from local path
local_model_path = "C:/Phi_models"

# Load Phi-3-mini model with float32 precision
tokenizer = AutoTokenizer.from_pretrained(local_model_path)
model = AutoModelForCausalLM.from_pretrained(
    local_model_path,
    device_map="cpu",  # Explicitly set to CPU
    torch_dtype=torch.float32,  # Force float32 to avoid Half precision error
    offload_folder="offload",  # Optional: folder for offloading weights
    trust_remote_code=True  # Required for some models
)

# Create text-generation pipeline
llm = pipeline("text-generation", model=model, tokenizer=tokenizer, device=-1)  # device=-1 for CPU

# === OCR TEXT INPUT ===
ocr_text = """
 গণপ্রজাতন্ত্রী বাংলাদেশ সরকার

7 Government of the People's Republic of Bangladesh

NATIONAL ID CARD / জাতীয় পরিচয় পত্র

নাম: মোঃ মুছা মন্ডল
Name: Md Mosa Mondol

পিতা; মোঃ গুনি মন্ডল!

মাতা: মোসাঃ চম্পা বেগম

al Date of Birth: 10 Mar 1987
a ID NO: 2617294213205
"""

# === PROMPT TEMPLATE ===
prompt = f"""
You are an expert at extracting structured NID card data.

Your task is to format the OCR result below into the following structure:
নাম: ...
Name: ...
পিতা: ...
মাতা: ...
স্বামী: Not Found
স্ত্রী: Not Found
Date of Birth: ...
ID NO: ...

If any field is missing, mark it as "Not Found". Only return the formatted output. Do not explain anything.

OCR TEXT:
{ocr_text}
"""

# Run generation
output = llm(prompt, max_new_tokens=300, do_sample=False)[0]["generated_text"]

# === Print final result (trim off prompt) ===
formatted_output = output.replace(prompt, "").strip()
print("\n📋 Formatted Result:\n")
print(formatted_output)