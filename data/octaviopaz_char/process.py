"""
Prepare the Shakespeare dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
Will save train.bin, val.bin containing the ids, and meta.pkl containing the
encoder and decoder and some other related info.
"""
import os
import pickle
import requests
import numpy as np

# Read the dataset
input_file_path = os.path.join(os.path.dirname(__file__), 'input_full.txt')
with open(input_file_path, 'r') as file:
    lines = file.readlines()

index = int(len(lines) * 0.05)

# Split the dataset into two parts
first_part = lines[:index]

# Save each half to a new file
output_file_path = os.path.join(os.path.dirname(__file__), 'output.txt')
print(output_file_path)
with open(output_file_path, 'w') as file:
    file.writelines(first_part)