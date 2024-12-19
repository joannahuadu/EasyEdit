# # Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6b", cache_dir="./hugging_cache")
# model = AutoModelForCausalLM.from_pretrained("EleutherAI/gpt-j-6b", cache_dir="./hugging_cache")
# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir="./hugging_cache")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B", cache_dir="./hugging_cache")

# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B", cache_dir="./hugging_cache")
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B", cache_dir="./hugging_cache")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5", cache_dir="/mnt/data2/workplace/data/hugging_cache")
# model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", cache_dir="/mnt/data2/workplace/data/hugging_cache")
tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")
model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5")


# Load model directly
# from transformers import AutoTokenizer, AutoModelForCausalLM
# from huggingface_hub import login
# login()

# tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B", cache_dir="/mnt/data2/workplace/data/hugging_cache")
# model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.1-8B", cache_dir="/mnt/data2/workplace/data/hugging_cache")