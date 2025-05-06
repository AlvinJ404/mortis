from transformers import AutoModelForCausalLM, AutoTokenizer
model_name = "microsoft/phi-3-mini-128k-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True)

input_text = "<|system|>\nYou are a helpful assistant.<|end|>\n<|user|>\nWhat is the capital of France?<|end|>\n<|assistant|>"
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(response)