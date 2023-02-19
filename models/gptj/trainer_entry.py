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

from pathlib import Path
from tqdm.notebook import tqdm#, range
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from trainer import *
from utils import *

if __name__=='__main__':
    args = parse_args()
    device = torch.device(args.device)

    # Create Dataset in Memory
    print('Processing Dataset')
    #  df_trn, df_tst = process_datasets(args.dataset_path, 7)
    #  df_trn, df_tst = prepare_discussion_dataset(args.dataset_path, 1024)
    df_trn, df_val = prepare_convo_dataset(args.dataset_path, 2048)

    #Loading the modles
    set_seed(args.seed)
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        filename='training.log',
                        level=logging.INFO 
                        )
    
    # Meep 
    print('Setting Up Tokenizers and (Possibly) PreTrained Models')

    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir = args.cache_dir)
    model = GPTJForCausalLM.from_pretrained("hivemind/gpt-j-6B-8bit", low_cpu_mem_usage=True)
    add_adapters(model)
    args.device = torch.device(args.device)
    model.to(args.device)
  
    # TODO Need To Deal with checkpoints
    
    # Do Training
    if args.do_training:
        train_dataset= ConvoDataset(tokenizer,args,df_trn,logger)
        val_dataset= ConvoDataset(tokenizer,args,df_trn,logger,"val_")

        print("Starting Training")
        global_step, tr_loss = train(args, train_dataset, val_dataset, model, tokenizer)
        logger.info("Global Step = %d, average loss = %s", global_step,tr_loss)
