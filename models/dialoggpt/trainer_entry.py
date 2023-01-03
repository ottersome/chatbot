import argparse
import logging
import os
from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import glob
import random
from typing import Dict, List, Tuple
import pandas as pd
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
    df_trn, df_tst = prepare_discussion_dataset(args.dataset_path, 1024)

    #Loading the modles
    set_seed(args.seed)
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        filename='training.log',
                        level=logging.INFO 
                        )
    print('Setting Up Tokenizers and (Possibly) PreTrained Models')
    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
    #  config.n_positions = 4096
    #  config.n_ctx= 4096
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, cache_dir = args.cache_dir)
    model = AutoModelWithLMHead.from_pretrained(
            args.model_name_or_path, 
            from_tf=False,
            config=config,
            cache_dir=args.cache_dir)

    model.to(device)
    # TODO Need To Deal with checkpoints
    
    # Do Training
    if args.do_training:
        train_dataset = DiscussionDataset(tokenizer,args,df_trn,logger)

        print("Starting Training")
        global_step, tr_los = train(args, train_dataset, model, tokenizer)
        logger.info("Global Step = %, average loss = %s", global_step,tr_loss)
