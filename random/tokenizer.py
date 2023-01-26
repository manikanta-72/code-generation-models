from transformers import RobertaTokenizer

text = "  reverse_element = sort(arr)"

tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')

input_ids = tokenizer(text,return_tensors="pt").input_ids

print(input_ids)
print(tokenizer.tokenize(text))
