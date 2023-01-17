import glob
import logging
import os
import pickle
import random
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd 
import numpy as np
import torch


from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm.notebook import tqdm, trange

from pathlib import Path

from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    AutoModelWithLMHead,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from utils import set_seed

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    from tensorboardX import SummaryWriter


# try:
    # from torch.utils.tensorboard import SummaryWriter
# except ImportError:
    # from tensorboardX import SummaryWriter

# Configs
logger = logging.getLogger(__name__)

MODEL_CONFIG_CLASSES = list(MODEL_WITH_LM_HEAD_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)

# TODO we must think about changing the way we process these prompt-> reponses
# I dont t like these proces of "adding context".


def train(args, train_dataset, val_dataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int,float]:

    # For main gpu 
    tb_writer = SummaryWriter()

    #Start Summary Writter
    # Pad Sequence
    batch_size = args.batch_size_per_gpu* args.n_gpu # Lets leave it at that for now 
    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    # Sample from training dataset
    print(f"Size of training data set is {len(train_dataset)}")
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset,sampler=train_sampler, batch_size=batch_size, collate_fn=collate, drop_last=True)

    # Start Model and resize token embeddings
    model.resize_token_embeddings(len(tokenizer))
    no_decay = ["bias","LayerNorm.weight"]
    optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": args.weight_decay,
            },
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
        ]

    #total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Chose Adam as optimizer
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # Scheduler with WarmpUp an then linear decrease
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
    )
    #
    # TODO: Check if checkopoints exists
    
    # Actually Train
        # Show some Logging Info 
        # Check for model checkpoint
        # Start with checkpointif available
        # Start Training Itaration

    if args.fp_16:
        try :
            from apex import amp 
        except ImportError:
            raise ImportError("need to isntall apex for nvidia to use fp16 training. ")
        model,optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    #TODO think about rank thing

    logger.info("***** Started training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    #logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info(
        "  Total train batch size (w. parallel, distributed & accumulation) = %d",
        args.batch_size_per_gpu
        # * args.gradient_accumulation_steps
        # * (torch.distributed.get_world_size() if args.local_rank != -1 else 1),
    )
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", total_training_steps)

    # TODO load checkpoints

    model.zero_grad() 
    train_iterator = trange(0,int(args.num_train_epochs),desc="Epoch")
    global_step = 0
    logging_loss, tr_loss = 0.0,0.0
    set_seed(args.seed)
    epoch_wise_loss = []
    epoch_wise_valloss = []


    epoch = 0
    for _ in train_iterator:
        within_epoch_iterator = tqdm(train_dataloader, desc="Iteration")
        print('At epoch:',epoch)
        epoch += 1
        for step, batch in enumerate(within_epoch_iterator):
            inputs, labels = (batch, batch)
            #  print("I will show you the input and the batch")
            #  print(batch)

            print("At step: ", step)
            # Skip Long Examples
            if inputs.shape[1] > 4096:
                print("Skipping this example")
                continue

            inputs = inputs.to(args.device)
            labels = labels.to(args.device)

            model.train()
            outputs  = model(inputs,labels=labels)

            loss = outputs[0]
            epoch_wise_loss.append(loss)

            if args.n_gpu > 1:
                loss = loss.mean()

            if args.fp_16:
                with amp.scale_loss(loss,optimizer) as scaled_loss:
                    scaled_loss.backward()
            else: 
                loss.backward()

            tr_loss += loss.item()

            # Here we might use accumulation steps
            optimizer.step()
            scheduler.step()
            model.zero_grad()
            global_step+=1

            print(f'Total loss so far : {tr_loss/global_step}')
            logger.info(f'Loss for epoch {epoch} is {tr_loss/global_step}')

            if global_step % args.logging_steps == 0:
                tb_writer.add_scalar("lr", scheduler.get_lr()[0],global_step)
                tb_writer.add_scalar("loss", (tr_loss - logging_loss)/args.logging_steps)
                
                logging_loss = tr_loss

            if args.max_steps > 0 and global_step > args.max_steps:
                epoch_iterator.close()
                break

        if global_step % args.checkpoint_interval == 0:
        #  if True:
            output_dir = os.path.join(args.output_dir,"{}-{}".format("chkpnt",global_step))
            os.makedirs(output_dir, exist_ok=True)
            model_to_save = model

            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir,"training_args.bin"))

            torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
            torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
            print(f'Sving checkpoint at global step {global_step}')
            logger.info("Saving optimizer and scheduler states to %s", output_dir)
            logger.info("\tLoss foar this training epoch was:%f", tr_loss/global_step)
        # For now we will evaluate on every epoch 
        if global_step% 1 == 0:
            val_score = evaluate(args, model, tokenizer, val_dataset)
            epoch_wise_valloss.append(val_score)
            # TODO compare validation results to see if there is no longer any improvement
            logger.info(f'Validation loss(perplexity) for epoch {epoch} is {val_score}')
            if len(epoch_wise_valloss) > 3 :
                threshold_crossed  = (epoch_wise_valloss[-1]-epoch_wise_valloss[-2] > 0 ) and (epoch_wise_valloss[-2]-epoch_wise_valloss[-3] > 0 )
                output_dir = os.path.join(args.output_dir,"{}-{}".format("early_stop",global_step))
                os.makedirs(output_dir,exist_ok=True)
                if threshold_crossed:
                    logger.info('We have detected an increase in validation loss in two consecutive epochs.')
                    logger.info('We will now save this model and stop trianing ')

                    torch.save(epoch_wise_valloss,os.path.join(output_dir,'valloss.pt'))
                    torch.save(epoch_wise_loss,os.path.join(output_dir,'loss.pt'))
                    torch.save(optimizer.state_dict(),os.path.join(output_dir,'optimizer.pt'))
                    torch.save(scheduler.state_dict(), os.path.join(output_dir, 'scheduler.pt'))
                    print(f'We have hit early stopping at epoch {epoch} saving and breaking now...')
                    
                    break
        if args.max_steps > 0 and global_step > args.max_steps:
            print("global steps has overcome max steps")
            train_iterator.close()
            break
    tb_writer.close()
    avg_loss = tr_loss /global_step
    print(f"Finished with training with global_step:{global_step} and average loss {avg_loss}")
    return global_step, avg_loss



def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix ="") -> Dict:
    # Create output dir
    #  eval_output_dir = args.output_dir
    #  os.makedirs(eval_output_dir, exist_ok=True)
    batch_size = args.batch_size_per_gpu* max(1, args.n_gpu)

    def collate(examples: List[torch.Tensor]):
        if tokenizer._pad_token is None:
            return pad_sequence(examples, batch_first=True)
        return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, collate_fn=collate, drop_last=True)

    # TODO set model for multi gpu environment
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    for batch in tqdm(eval_dataloader):
        inputs, labels = (batch,batch)
        inputs = inputs.to(args.device)
        labels = labels.to(args.device)
        with torch.no_grad():
            outputs = model(inputs, labels=labels)
            lm_loss = outputs[0]
            eval_loss += lm_loss.mean().item()
        nb_eval_steps += 1

    eval_loss = eval_loss / nb_eval_steps
    perplexity = torch.exp(torch.tensor(eval_loss))

    return perplexity
