from deepfloyd_if.modules.t5 import T5Embedder
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download
import os

dir_or_name = 't5-v1_1-xxl'
cache_dir = os.path.expanduser('~/.cache/IF_')
hf_token= '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'
cache_dir = os.path.join(cache_dir, dir_or_name)

for filename in [
    'config.json', 'special_tokens_map.json', 'spiece.model', 'tokenizer_config.json',
    'pytorch_model.bin.index.json', 'pytorch_model-00001-of-00002.bin', 'pytorch_model-00002-of-00002.bin'
]:
    hf_hub_download(repo_id=f'DeepFloyd/{dir_or_name}', filename=filename, cache_dir=cache_dir,
                    force_filename=filename, token=hf_token)
    
tokenizer_path = cache_dir
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

texts = 'photo of a cat'

text_tokens_and_mask = tokenizer(
    texts,
    max_length=77,
    padding='max_length',
    truncation=True,
    return_attention_mask=True,
    add_special_tokens=True,
    return_tensors='pt'
)

# 查看字典中的键
print("Keys in text_tokens_and_mask:", text_tokens_and_mask.keys())

# 查看变量的类型
print("Type of text_tokens_and_mask:", type(text_tokens_and_mask))

# 查看内容
print("tokens:")
print(text_tokens_and_mask['input_ids'])

# 查看对应关系
words_to_tokens = []

#BatchEncoding.word_ids returns a list mapping words to tokens
for w_idx in set(text_tokens_and_mask.word_ids()):
    if w_idx==None:
        continue

    #BatchEncoding.word_to_tokens tells us which and how many tokens are used for the specific word
    start, end = text_tokens_and_mask.word_to_tokens(w_idx)
    words_to_tokens.append(list(range(start,end)))

print("words:", texts)
print("words to tokens:", words_to_tokens)