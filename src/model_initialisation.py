import os
import torch
import transformers


def init_model(cuda_available: bool):
    hf_token = os.getenv("hf_token")
    if not hf_token:
        raise RuntimeError("hf_token not found in .env")
    llm_name = os.getenv("llm_name")
    
    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_config = transformers.AutoConfig.from_pretrained(
        pretrained_model_name_or_path=llm_name,
        token=hf_token
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=llm_name,
        config=model_config,
        quantization_config=bnb_config if cuda_available else None,
        device_map="auto" if cuda_available else "cpu",
        token=hf_token
    )

    model.eval()

    return model, llm_name
