import json
import ast
from pathlib import Path
from pprint import pprint
import libcst as cst

with open('/home/mani/git/CodeSearchNet/resources/data/python/final/jsonl/train/python_train_0.jsonl','r') as f:
    sample_file = f.readlines()
print(sample_file[0])

# pprint(json.loads(sample_file[0]))

json_data = json.loads(sample_file[0])
print(json_data['original_string'])
code = json_data['original_string']

# pprint(ast.dump(ast.parse(code)))


names = [
    node.id for node in ast.walk(ast.parse(code))
    if isinstance(node, ast.Name)
]

# print(names)

text = '''# This program adds two numbers

num1 = 1.5
num2 = 6.3

# Add two numbers
sum = num1 + num2
count = 0
for i in range(10):
    count += i
    print(count)
# Display the sum
print('The sum of {0} and {1} is {2}'.format(num1, num2, sum))
'''

pprint(ast.dump(ast.parse(text), indent=4))

names = [
    node.id for node in ast.walk(ast.parse(text))
    if isinstance(node, ast.Name) and node._fields is not 'func'
]

fields = [
        node._fields for node in ast.walk(ast.parse(text))
]
print(names)
print("fields:", fields)

names = [c for c in ast.iter_fields(ast.parse(text))]

print(names)

# print(cst.parse_module(text))
