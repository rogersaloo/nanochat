from collections import deque

import torch

from nanochat.common import get_dist_info
from nanochat.dataset import parquets_iter_batched
from nanochat.tokenizer import get_tokenizer

def tokenizing_distributed_data_loader(B, T, split, tokenizer_threads=4, tokenizer_batch_size=128):
    """ Steaming pretraining text from parquet files. tokenize, yield training batches"""
    assert split in ["train", "val"], "split must be teain or val"
    ddp, ddp_rank, ddp_local_rank, ddp_world_size = get_dist_info()
    needed_tokens = B * T + 1
    # get the tokenizer and the bos token 
    tokenizer = get_tokenizer()
    bos_token = tokenizer.get_bos_token_id()
    # scratch buffer hold the tokens for one iteration
    token_buffer = deque()
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=True)
    
    #infinte iterator over document batches
    def document_batches():
        while True:
            # batch will iterate the gtoup size of the paraquet file usually 1024
            for batch in parquets_iter_batched(split=split, start=ddp_rank, step=ddp_world_size):
                # for the tokenizer we proceed with smaller batches ie 128
                for i in range(0, len(batch), tokenizer_batch_size):
                    yield batch[i:i+tokenizer_batch_size]
    batches = document_batches()
    
    batch_index = 0
    while True:
        #Accumumate enough tokens for iteration before yielding. 
        while len(token_buffer) < needed_tokens:
            doc_batch = next(batches)
            token_lists = tokenizer.encode(doc_batch, prepend=bos_token, num_threads=tokenizer_threads)
            for tokens in token_lists:
                token_buffer.extend(tokens)
            batch_index += 1
            # Move tokens from the deque into the scratch buffer
            for i in range(needed_tokens):
                scratch[i] = token_buffer.popleft()
            # create the inputs/targets as 1D tensor
            inputs_cpu = scratch[:-1].to(dtype=torch.int32)
            targets_cpu = scratch[1:]
            # reshape to 2D and move to the GPU
            inputs = inputs_cpu.view(B, T).to(device="cuda", dtype=torch.int32, non_blocking=True)
            targets = targets_cpu.view(B, T).to(device="cuda", dtype=torch.int32, non_blocking=True)
            yield inputs, targets
            