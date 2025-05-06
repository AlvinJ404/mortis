from modelscope import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen3-8B", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen3-8B", device_map="auto", trust_remote_code=True)

input_text = "9.9和9.11谁大？"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output = model.generate(input_ids, max_new_tokens=50)
print(tokenizer.decode(output[0], skip_special_tokens=True))
