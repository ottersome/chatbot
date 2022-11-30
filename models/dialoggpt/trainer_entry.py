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


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if(torch.cuda.device_count() > 0):
        torch.cuda.manual_seed_all(seed)


if __name__=='__main__':
    args = parse_args()
    device = torch.device(args.device)

    # Create Dataset in Memory
    print('Processing Dataset')
    df_trn, df_tst = process_datasets(args.dataset_path, 7)

    #Loading teh modles
    set_seed(args.seed)
    
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        level=logging.INFO 
                        )
    config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
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
        train_dataset = DialogDataset(tokenizer,args,df_trn,logger)

        global_step, tr_los = train(args, train_dataset, model, tokenizer)
        logger.info("Global Step = %, average loss = %s", global_step,tr_loss)
