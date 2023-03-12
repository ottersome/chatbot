import argparse
import logging
import os
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import glob
import random
from typing import Dict, List, Tuple
import pandas as pd
from GPTJ8bit import *
import numpy as np
from torch.nn.utils.rnn import pad_sequence

import torch.distributed as dist

from pathlib import Path
from tqdm.notebook import tqdm#, range
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from trainer import *
from utils import *
from datasets import *

def cross_process_setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # initialize the process group
    dist.init_process_group("gloo", rank=rank, world_size=world_size)

def cleanup():
    dist.destroy_process_group()

def per_process_setup(rank, world_size):
    cross_process_setup(rank,world_size)

    model = ToyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])



if __name__=='__main__':
    args = parse_args()
    device = torch.device(args.device)

    # Create Dataset in Memory
    print('Processing Dataset')
    # df_trn, df_val = prepare_convo_dataset(args.dataset_path, 2048)

    #Loading the modles
    set_seed(args.seed)
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        filename='training.log',
                        level=logging.DEBUG
                        )
    
    # Meep 
    print('Setting Up Tokenizers and (Possibly) PreTrained Models')

    #config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir = args.cache_dir)
    model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
    add_adapters(model)
    model.gradient_checkpointing_enable()
    # Load Checkpoint
    if args.checkpoint_path != "":
        model.load_state_dict(torch.load(args.checkpoint_path+'/model_state_dict.pt'))

    args.device = torch.device(args.device)
    model.to(args.device)
    model = nn.DataParallel(model)
  
    # Crearte Output dir
    p=Path(args.output_dir)
    p.mkdir(parents=True, exist_ok=True)
    
    # Do Training
    if args.do_training:
        bds = BotDataset(tokenizer,args,logger)

        print("Starting Training")
        global_step, tr_loss = train(args, bds, model, tokenizer)
        logger.info("Global Step = %d, average loss = %s", global_step,tr_loss)
