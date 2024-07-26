from modeling_llamagear import LlamaForCausalLM_GEARKIVI

from modeling_llama_kivi import LlamaForCausalLM_KIVI
from transformers import LlamaConfig, AutoTokenizer, LlamaForCausalLM
from transformers import BitsAndBytesConfig
from datasets import load_dataset
import torch
import argparse


config = LlamaConfig.from_pretrained("meta-llama/Llama-2-7b-hf")

config.k_bits = 2# current support 2/4 bit for KV Cache
config.v_bits = 2 # current support 2/4 bit for KV Cache
config.group_size = 64
config.residual_length = 64 # the number of recent fp16 tokens

# quantization_config = BitsAndBytesConfig(load_in_8bit=True)
parser = argparse.ArgumentParser(description="Evaluate AQuA Tasks")
parser.add_argument("--batch_size", type=int, default=4, help="Batch size.")
# parser.add_argument("--model", type=str, default="meta-llama/Llama-2-7b", help="Model name or path.")
parser.add_argument("--model", type=str, default="None", help="Model name or path.")
args = parser.parse_args()

max_token = 1000 ### prefill_length
max_generation_length = 1500 ### geneate 500
batch_size = args.batch_size

##### Config for 
compress_config = {}
compress_config["compress_method"] = "gearlKIVI" # "gearlKIVI" "gearsKIVI"
compress_config["group_size"] = 64
compress_config["residual"] = 64
compress_config["quantize_bit"] = 2
compress_config["rank"] = 2 ## prefill rank
compress_config["rankv"] = 2 ## prefill rank
compress_config["loop"] = 3
stream_list = [torch.cuda.Stream(),torch.cuda.Stream()]
# compress_config["stream_list"] = stream_list

model = LlamaForCausalLM_GEARKIVI.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    config = config,
    # quantization_config = quantization_config,
    compress_config = compress_config,
    torch_dtype=torch.float16, # FP16 으로 불러오도록 추가됨.
    device_map = "cuda:0"
)

model = model.half()

tokenizer = AutoTokenizer.from_pretrained(
    'meta-llama/Llama-2-7b-hf', 
    model_max_length=max_token,
    max_length=max_token,
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')
tokenizer.pad_token = tokenizer.eos_token
test = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
text_combined = test["text"]

sentence_group = []
for i in range(batch_size):
    # sentence_group.append(str(text_combined[i*max_token:(i+1)*max_token]))
    sentence_group.append(str(text_combined[0:max_token]))
inputs = tokenizer(
    sentence_group,
    return_tensors="pt",
    padding="max_length",
    truncation=True,
)
print("begin")
inputs = inputs.to("cuda:0")
print(inputs.input_ids.shape)
import time

start = time.time()
result = model.generate(**inputs, max_length=max_generation_length, use_cache=True)
torch.cuda.synchronize()
end = time.time()
peak_memory = torch.cuda.max_memory_allocated(device="cuda") / (1024**2) 

print(f"Peak memory usage on GPU: {peak_memory} MB")
print("time",end - start)





# import torch
# from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding

# rotary_emb = LlamaRotaryEmbedding(
#                 128,
#                 max_position_embeddings=4096,
#                 base=10000.0,
#             ).to("cuda:0")
# dummy_value_states = torch.randn((32, 1000, 128), dtype=torch.float16).to("cuda:0")
# dummy_position_ids = torch.arange(dummy_value_states.size(1)).unsqueeze(0).to("cuda:0")
# cos, sin = rotary_emb(dummy_value_states, seq_len=1000, position_ids=dummy_position_ids)
# print(cos, sin)