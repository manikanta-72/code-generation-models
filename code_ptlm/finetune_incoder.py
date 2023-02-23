import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM
# import datasets
from  datasets  import  load_dataset, load_from_disk

model_name = "facebook/incoder-6B"
CACHE_DIR = "./cache/"

def load_dataset(search_patter: str):
    ds = load_dataset("bigcode/the-stack", data_dir="data/python", split="train")
    search_pattern = "Crypto"
    contains_crypto = ds.filter(lambda ex: True if search_pattern in ex['content'] else False)
    print(contains_crypto[0]['content'].split(' '))


def generate_number_of_spans(mean: torch.float32) -> int:
    return torch.clamp(torch.poisson(mean), 1, 256)

def overlap_checker(span_windows: list, start: int, end: int) -> tuple:
    if len(span_windows) == 0:
        return True, 0
    if span_windows[len(span_windows)-1][1] < start:
        return True, len(span_windows)
    elif span_windows[0][0] > end :
        return True, 0
    l, r = 0, len(span_windows)-1
    m = 0
    while(l<=r):
        m = l + (r-l)//2
        if span_windows[m][0] == start:
            return False, 0
        if span_windows[m][0] < start:
            l = m + 1
        else:
            r = m - 1
    if r+1 == len(span_windows):
        if span_windows[r][1] < start: 
            return True, r+1
        else:
            return False, 0
    if span_windows[r+1][0] == start:
        return False, 0
    elif span_windows[r+1][0] < end or span_windows[r][1] == start:  
        return False, 0
    return True, r+1

def generate_span_windows(doc_len: torch.float32, number_of_spans: int) -> list:
    span_windows = list()
    while(len(span_windows) < number_of_spans+1):
        uniform_sampler = torch.distributions.Categorical([1/doc_len]*doc_len)
        span_start = uniform_sampler.sample()
        span_end = uniform_sampler.sample()
        span_start, span_end = span_start, span_end if span_start <= span_end else span_end,span_start
        valid, index = overlap_checker(span_windows, span_start, span_end)
        if valid == True:
            span_windows.insert(index, [span_start,span_end])
    
    return span_windows

def make_sentinel(i:int) -> str:
    # signals (1) a location to insert an infill and (2) the start of the infill generation
    return f"<|mask:{i}|>"

def tokenizer_function(x):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer(x['content'], padding="max_length", truncation=True)

def main():
    # load datasets
    # From local_folder
    # x_train = load_from_disk("./datasets/adv_train")
    # x_test = load_from_disk("./datasets/adv_train")
    
    # from huggingFace 
    x_train = load_dataset("bigcode/the-stack", data_dir="data/python", split="train")
    # deduplication need to be done for datasets for creating train-test split.
    x_test = load_dataset("bigcode/the-stack", data_dir="data/python", split="val")

    config_kwargs = {
        "cache_dir": CACHE_DIR,
        "revision":"main",
        }

    config = AutoConfig.from_pretrained(model_name, **config_kwargs)

    kwargs = dict(
        revision="main", 
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        cache_dir = CACHE_DIR,
    )

    # load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        **kwargs)

    x_train_tokenized = x_train.map(tokenizer_function, batched=True)
    x_test_tokenized = x_test.map(tokenizer_function, batched=True)
    

if __name__ == "__main__":
    main()