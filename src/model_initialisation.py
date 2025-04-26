import torch
import transformers


def init_model(hf_token: str, cuda_available: bool):
    #model_name = "mistralai/Mistral-7B-Instruct-v0.2"
    model_name = "openai-community/gpt2"

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,  # loading in 4 bit
        bnb_4bit_quant_type="nf4",  # quantization type
        bnb_4bit_use_double_quant=True,  # nested quantization
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model_config = transformers.AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model_name,
        token=hf_token
    )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_name,
        config=model_config,
        quantization_config=bnb_config if cuda_available else None,
        device_map="auto" if cuda_available else "cpu",
        token=hf_token
    )

    model.eval()

    return model, model_name
