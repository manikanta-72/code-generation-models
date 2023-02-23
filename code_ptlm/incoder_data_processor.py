import torch
import transformers

# select the data samples that have tokens < 2048 
# as the model in token generation
def is_token_length_valid(x :torch.tensor, tokenizer_function: function) -> bool:
    x_tokenized = tokenizer_function(x)
    return len(x_tokenized) > 2048 - 128

# parse and remove unwanted foreign tokens

# 