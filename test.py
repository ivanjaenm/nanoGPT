"""
Test script for generating text from a trained model.
"""
import os
import pickle
from contextlib import nullcontext
import torch
import tiktoken
import numpy as np
from model import GPTConfig, GPT
from torch.nn import functional as F
from collections import Counter
import matplotlib.pyplot as plt
from scipy.special import rel_entr

# -----------------------------------------------------------------------------
init_from = 'resume' # either 'resume' (from an out_dir) or a gpt2 variant (e.g. 'gpt2-xl')
dataset = "octaviopaz_char" # dataset name
#dataset = "shakespeare_char" # dataset name
char_length = '95k' # number of characters in the dataset
input_filename = 'input_{}.txt'.format(char_length)
input_filename_val   = 'val_{}.bin'.format(char_length)
input_filename_meta  = 'meta_{}.pkl'.format(char_length)
input_filename_checkpoint = 'ckpt_{}.pt'.format(char_length)

out_dir = 'out' # ignored if init_from is not 'resume'
start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
num_samples = 40 # number of samples to draw
max_new_tokens = 500 # number of tokens generated in each sample
temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
compile = False # use PyTorch 2.0 to compile the model to be faster
exec(open('configurator.py').read()) # overrides from command line or config file
# -----------------------------------------------------------------------------

# Evaluation Metrics
# General metric: Perplexity
# https://en.wikipedia.org/wiki/Perplexity
def compute_eval_perplexity(model, device, num_batches):
    model.eval()
    total_loss = 0
    total_count = 0

    with torch.no_grad():
        for _ in range(num_batches):
            inputs, targets = get_val_batch()
            with ctx:
                logits, loss = model(inputs, targets)
            log_probs = F.log_softmax(logits, dim=-1)
            
            # Reshape log_probs and targets for calculating loss
            # Flatten targets to [batch_size * sequence_length]
            targets = targets.view(-1)
            # Flatten log_probs to [batch_size * sequence_length, vocab_size]
            log_probs = log_probs.view(-1, log_probs.size(-1))

            # Calculate the loss
            loss = F.nll_loss(log_probs, targets, reduction='sum')
            
            total_loss += loss.item()
            total_count += targets.numel()  # Count total number of target tokens

    average_loss = total_loss / total_count
    perplexity = np.exp(average_loss)
    return perplexity

# Specific Metric: KLD (Kullback-Leibler Divergence) comparing training and test distributions
#https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence

# Read the dataset
def get_token_distribution(dataset):    
    with open(dataset, 'r') as f:
        data = f.read()
    
    # Count the frequency of each character in the dataset
    char_counts = Counter(data)

    # Calculate the total number of characters in the dataset
    total_chars = sum(char_counts.values())

    # Normalize the frequencies to get the discrete probability distribution
    char_distribution = {char: count / total_chars for char, count in char_counts.items()}

    # Print the distribution
    # print("\nCharacter Distribution:")
    # for char, prob in sorted(char_distribution.items(), key=lambda x: -x[1]):
    #     print(f"'{char}': {prob:.5f}")

    return char_distribution

def plot_distribution(char_distribution, dataset):

    # Plot the distribution of the most common characters
    sorted_chars = sorted(char_distribution.items(), key=lambda x: -x[1])

    chars, probs = zip(*sorted_chars)

    plt.figure(figsize=(10, 6))
    plt.bar(chars[:50], probs[:50])  # Visualize the top 50 characters
    plt.xlabel("Characters")
    plt.ylabel("Probability")
    plt.title("Character Distribution in {} Dataset".format(dataset))
    plt.show()

# Function to calculate KL divergence between two distributions
def compute_eval_kl_divergence(pred_dist, ref_dist):
    # Convert the distributions to match the same vocabulary space
    all_tokens = set(pred_dist.keys()).union(set(ref_dist.keys()))
    
    # Fill missing tokens with a small probability to avoid zero division
    epsilon = 1e-10
    pred_probs = np.array([pred_dist.get(token, epsilon) for token in all_tokens])
    ref_probs = np.array([ref_dist.get(token, epsilon) for token in all_tokens])
    
    # Calculate KL divergence
    kl_divergence = np.sum(rel_entr(pred_probs, ref_probs))
    return kl_divergence


torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


# Load the validation data
data_dir = os.path.join('data', dataset)
val_data = np.memmap(os.path.join(data_dir, input_filename_val), dtype=np.uint16, mode='r')
print(data_dir)

# Batch generation function for the validation set
def get_val_batch():
    ix = torch.randint(len(val_data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((val_data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((val_data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    if device_type == 'cuda':
        # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    else:
        x, y = x.to(device), y.to(device)
    return x, y

# model
if init_from == 'resume':
    # init from a model saved in a specific directory
    ckpt_path = os.path.join(out_dir, input_filename_checkpoint)
    print("Loading model from: ", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    gptconf = GPTConfig(**checkpoint['model_args'])
    model = GPT(gptconf)
    state_dict = checkpoint['model']
    unwanted_prefix = '_orig_mod.'
    for k,v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
    model.load_state_dict(state_dict)
elif init_from.startswith('gpt2'):
    # init from a given GPT-2 model
    model = GPT.from_pretrained(init_from, dict(dropout=0.0))

model.eval()
model.to(device)
if compile:
    model = torch.compile(model) # requires PyTorch 2.0 (optional)

# Testing Configuration

# Evaluation metric: KLD (Kullback-Leibler Divergence)
generated_samples = []
# look for the meta pickle in case it is available in the dataset folder
load_meta = False
if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
    meta_path = os.path.join('data', checkpoint['config']['dataset'], input_filename_meta)
    load_meta = os.path.exists(meta_path)
if load_meta:
    print(f"Loading meta from {meta_path}...")
    with open(meta_path, 'rb') as f:
        meta = pickle.load(f)
    # TODO want to make this more general to arbitrary encoder/decoder schemes
    stoi, itos = meta['stoi'], meta['itos']
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: ''.join([itos[i] for i in l])
else:
    # ok let's assume gpt-2 encodings by default
    print("No meta.pkl found, assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

# encode the beginning of the prompt
if start.startswith('FILE:'):
    with open(start[5:], 'r', encoding='utf-8') as f:
        start = f.read()
print(f"Prompting with: {start}")
start_ids = encode(start)
x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
# run generation
print("Starting generation")
with torch.no_grad():
    with ctx:
        for k in range(num_samples):
            y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
            sample = decode(y[0].tolist())
            generated_samples.append(sample)       
            print(sample)

# generated data distribution
single_string = ' '.join(generated_samples)
output_dataset_path = os.path.join('data', dataset + "/" + 'output.txt')
# Write the single string to the file
with open(output_dataset_path, 'w') as file:
    file.write(single_string)
generated_dist = get_token_distribution(output_dataset_path)
#plot_distribution(generated_dist, "Generated")

# training distribution
training_dataset_path = os.path.join('data', dataset + "/" + input_filename)
training_dist = get_token_distribution(training_dataset_path)
#plot_distribution(training_dist, "Training")

kld = compute_eval_kl_divergence(generated_dist, training_dist)
print(f"Kullback-Leibler Divergence: {kld}")

# Evaluation metric: Perplexity
block_size = 64  
batch_size = 12
num_batches = 20
perplexity = compute_eval_perplexity(model, device, num_batches)
print(f"Perplexity: {perplexity}")
