from transformers import AutoModelForCausalLM, BitsAndBytesConfig

# model was downloaded from https://huggingface.co/cyberagent/calm3-22b-chat
quantization_config = BitsAndBytesConfig(load_in_4bit=True)
model = AutoModelForCausalLM.from_pretrained(
    "cyberagent/calm3-22b-chat",
    quantization_config=quantization_config
)
model.save_pretrained("calm3-22b-chat-4bit")