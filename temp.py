from deepfloyd_if.modules.t5 import T5Embedder
from transformers import AutoTokenizer

tokenizer_path = 't5-v1_1-xxl'
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