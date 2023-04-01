import glob
import logging
import os
import pickle
import random
import sys
import re
import shutil
from typing import Dict, List, Tuple

import pandas as pd 
import numpy as np
import torch
import bitsandbytes as bnb
import torch.nn.functional as F

from datasets import *
# from pytorch_memlab import LineProfiler, MemReporter
# from GPUtil import showUtilization as gpu_usage

from torch.nn.utils.rnn import pad_sequence
from torch.nn.functional import pad, logsigmoid
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
import gc

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
from utils import set_seed,get_mask

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

def save_checkpoint(model, optimizer, args,tinfo):
    output_dir = os.path.join(args.output_dir,"{}-{}-{}".format("chkpnt",tinfo['epoch'],tinfo['global_step']))
    os.makedirs(output_dir, exist_ok=True)
    logger.info("Saving model on the {}-th epoch and {}-th global step into {}".format(tinfo['epoch'],tinfo['global_step'], output_dir))

    torch.save(model.module.state_dict(),os.path.join(output_dir,"model_state_dict.pt"))
    torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
    torch.save(tinfo, os.path.join(output_dir, "tinfo.bin"))
    

def train(args, dataset: BinaryFeedbackDataset, model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> Tuple[int,float]:

    def collate(examples: List[torch.Tensor]):
        logger.info("Size of examples here is :"+str(len(examples)))
        gcombos,bcombos = [],[]
        gtids,btids= [],[]
        for example in examples:
            gcombos.append(example['gcombo'])
            bcombos.append(example['bcombo'])
            gtids.append(example['good_type_ids'])
            btids.append(example['bad_type_ids'])


        if tokenizer._pad_token is None:
            # Will likey by this with GPTJ, so by default it will pad with 0
            g_padded = pad_sequence(gcombos, batch_first=True)
            b_padded = pad_sequence(bcombos, batch_first=True)
            gtids_padded = pad_sequence(gtids, batch_first=True)
            btids_padded = pad_sequence(btids, batch_first=True)
            return g_padded,b_padded, gtids_padded, btids_padded

        g_padded = pad_sequence(gcombos, batch_first=True, padding_value=tokenizer.pad_token_id)
        b_padded = pad_sequence(bcombos, batch_first=True, padding_value=tokenizer.pad_token_id)
        gtids_padded = pad_sequence(gtids, batch_first=True, padding_value=tokenizer.pad_token_id)
        btids_padded = pad_sequence(btids, batch_first=True, padding_value=tokenizer.pad_token_id)
        return g_padded,b_padded, gtids_padded, btids_padded
    
    # no_decay = ["bias", "LayerNorm.weight"]
    # optimizer_grouped_parameters = [
           # {
               # "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
               # "weight_decay": args.weight_decay,
           # },
           # {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
       # ]

    # Choose Adam as optimizer
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate, eps=2)
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
           {
               "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
               "weight_decay": args.weight_decay,
           },
           {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},
       ]

    # Choose Adam as optimizer
    #optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    #  optimizer = bnb.optim.Adam8bit(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    tinfo = {"loss" : 0,"epoch": 1,"global_step" : 0 , "saved_step" : 0,"epoch_wise_loss" : [], "epoch_wise_valloss" : []}
    ########################################
    # Load Checkpoint
    ########################################
    #if args.checkpoint_path != "":
    if False: 
        print("Starting with checkpoint: "+args.checkpoint_path)
        optimizer.load_state_dict(torch.load(args.checkpoint_path+'/optimizer.pt'),strict=False)
        tinfo = torch.load(args.checkpoint_path+'/tinfo.bin')
        tinfo['epoch'] += 1 # Add a 1. Because it remembers last epoch not next one

    # For main gpu 
    tb_writer = SummaryWriter()

    #Start Summary Writter
    # Pad Sequence
    batch_size = args.batch_size_per_gpu# Lets leave it at that for now 
    
    # Here each example is made of a good and bad combination

    # Sample from training dataset
    print(f"Size of training data set is {len(dataset)}")
    train_sampler = RandomSampler(dataset)
    train_dataloader = DataLoader(dataset,sampler=train_sampler, batch_size=batch_size, collate_fn=collate, drop_last=True)

    # Start Model and resize token embeddings
    #model.resize_token_embeddings(len(tokenizer))
    # no_decay = ["bias","LayerNorm.weight"]


    # TODO: Maybe do gradient accumulation again?
    #total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs
    total_training_steps = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs


    # scheduler = get_linear_schedule_with_warmup(
        # optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_training_steps
    # )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_train_epochs)

    logger.info("***** Started training *****")
    logger.info("  Num examples = %d", len(dataset))
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


    print("Starting with epoch " ,tinfo['epoch'])
    model.zero_grad() 
    train_iterator = trange(tinfo['epoch'],int(args.num_train_epochs),desc="Epoch")
    tinfo['global_step'] = 0
    tr_loss = 0.0
    set_seed(args.seed)
    epoch_wise_valloss = tinfo['epoch_wise_valloss']
    

    logger.info("We will do savings per batch : {}".format(int(0.1*(dataset.len()/args.batch_size_per_gpu))))

    for _ in train_iterator:
        within_epoch_iterator = tqdm(train_dataloader, initial=tinfo['saved_step'],desc="Iteration", leave=False)
        tinfo['epoch'] += 1
        for batch in within_epoch_iterator:
            # <COmbos are all properly context+response>
            optimizer.zero_grad()
            gcombo, bcombo, gtypeids, btypeids =  batch

            gmasks, bmasks = get_mask(gcombo,0,tokenizer.eos_token_id), get_mask(bcombo,0,tokenizer.eos_token_id)

            gcombo = gcombo.to(args.device)
            bcombo = bcombo.to(args.device)
            gmasks = gmasks.to(args.device)
            bmasks = bmasks.to(args.device)
            gtypeids = gtypeids.to(args.device)
            btypeids = btypeids.to(args.device)

            # Skip Long Examples
            # if inputs.shape[1] > 4096:
                # print("Skipping this example")
                # continue


            model.train()

            #outputs  = model(inputs,labels=labels) # THis is for useing the internal loss function
            # TODO Maks for padding the batch 
            logger.info("Length of gcombo: {} and bcombo {}".format(gcombo.shape, bcombo.shape) )
            # extra_zeros = torch.tensor([[0]]*inputs.shape[0]).to(args.device)
            # labels = torch.cat([inputs[:,1:], extra_zeros],axis=1)
            # Labels can be set = inputs segun la documentacion
            # logger.debug("GCombo[0]  for thistep :\n{}".format(tokenizer.decode(gcombo[0])))
            gscore  = model(input_ids = gcombo,attention_mask= gmasks, token_type_ids=gtypeids, use_cache=False)
            # logger.debug("BCombo[0]  for thistep :\n{}".format(tokenizer.decode(bcombo[0])))
            bscore  = model(input_ids = bcombo,attention_mask= bmasks, token_type_ids=btypeids, use_cache=False)

            # Our own Loss
            #  loss = -torch.log(torch.sigmoid(gscore.logits - bscore.logits))
            #  loss = loss.mean()
            loss = -logsigmoid(gscore.logits - bscore.logits).mean()
            #  margin = 1.0  # You can set the margin to a suitable value
            #  loss = torch.clamp(margin - (gscore.logits - bscore.logits), min=0).mean()


            # Calculate Gradients
            loss.backward()

            #print("Scores are : g:{}\n\t and b:{}".format(gscore.logits.item(),bscore.logits.item()))
            print("Scores are : g:{}\n\t and b:{}".format(gscore.logits.mean().item(),bscore.logits.mean().item()))
            print("Learning Rates is : ", scheduler.get_lr())
            print("Transformer Block Weights Norm:", model.module.transformer.h[27].attn.q_proj.adapter[0].weight.abs().sum())
            print("Transformer Block Weights Grad:", model.module.transformer.h[27].attn.q_proj.adapter[0].weight.grad.abs().sum())
            print("Value  Weights Norm:", model.module.val_head.weight.abs().sum())
            #loss = outputs[0]
            # loss = F.cross_entropy(out.logits[:,:-1,:].flatten(0,-2), labels,reduction='mean')
            tinfo['epoch_wise_loss'].append(loss.item())

            tr_loss += loss.item()

            # Here we might use accumulation steps
            #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            tinfo['global_step']+=1
            
            # Log If Weights are changing
            #  print("Embedding Weights:", model.transformer.wte.adapter[1].weight)


            logger.info(f"Loss for epoch {tinfo['epoch']}, global steo {tinfo['global_step']} is {tr_loss/tinfo['global_step']}")
            within_epoch_iterator.set_description('CSLoss: {} CBLoss: {}'.format(loss.item(),tr_loss/tinfo['global_step']))

            del gcombo,bcombo, gmasks, gtypeids,btypeids,bmasks, gscore, loss, batch
            torch.cuda.empty_cache()
            gc.collect()
            tinfo['saved_step'] +=1
            ### END OF BATCH ##

            if tinfo['global_step'] % int(0.1*(dataset.len()/args.batch_size_per_gpu)) == 0:
                save_checkpoint(model, optimizer,args,tinfo)
        
        tinfo['saved_step'] =0
        ########################################
        # End of Epoch Maintenance
        ########################################
        if tinfo['epoch']  % args.checkpoint_interval:
            save_checkpoint(model, optimizer,args,tinfo)
        # Test
        dataset.change_mode(0)
        val_score = evaluate(args, model, tokenizer, dataset)
        tinfo['epoch_wise_valloss'].append(val_score)
        # TODO compare validation results to see if there is no longer any improvement
        logger.info(f"Validation loss(perplexity) for epoch {tinfo['epoch']} is {val_score}")
        if len(tinfo['epoch_wise_vallos']) > 3 :
            threshold_crossed  = (tinfo['epoch_wise_valloss'][-1]-tinfo['epoch_wise_valloss'][-2] > 0 ) \
                    and (tinfo['epoch_wise_valloss'][-2]-tinfo['epoch_wise_valloss'][-3] > 0 )
            if threshold_crossed:
                output_dir = os.path.join(args.output_dir,"{}-{}".format("early_stop",tinfo['global_step']))
                os.makedirs(output_dir,exist_ok=True)
                logger.info('We have detected an increase in validation loss in two consecutive epochs.')
                logger.info('We will now save this model and stop trianing ')

                save_checkpoint(model, optimizer,args, tinfo)
                torch.save(tinfo['epoch_wise_valloss'],os.path.join(output_dir,'valloss.pt'))
                torch.save(tinfo['epoch_wise_loss'],os.path.join(output_dir,'loss.pt'))
                torch.save(optimizer.state_dict(),os.path.join(output_dir,'optimizer.pt'))
                print(f'We have hit early stopping at epoch {tinfo["epoch"]} saving and breaking now...')
                
                break
        dataset.change_mode(1)

    train_iterator.close()
    tb_writer.close()
    avg_loss = tr_loss /tinfo['global_step']
    print(f"Finished with training with global_step:{global_step} and average loss {avg_loss}")
    return global_step, avg_loss



def evaluate(args, model: PreTrainedModel, tokenizer: PreTrainedTokenizer, eval_dataset, prefix ="") -> Dict:

    # Create output dir
    #  eval_output_dir = args.output_dir
    #  os.makedirs(eval_output_dir, exist_ok=True)
    def collate(examples: List[torch.Tensor]):
        logger.info("Size of examples here is :"+str(len(examples)))
        gcombos,bcombos = [],[]
        gtids,btids= [],[]
        for example in examples:
            gcombos.append(example['gcombo'])
            bcombos.append(example['bcombo'])
            gtids.append(example['good_type_ids'])
            btids.append(example['bad_type_ids'])


        if tokenizer._pad_token is None:
            # Will likey by this with GPTJ, so by default it will pad with 0
            g_padded = pad_sequence(gcombos, batch_first=True)
            b_padded = pad_sequence(bcombos, batch_first=True)
            gtids_padded = pad_sequence(gtids, batch_first=True)
            btids_padded = pad_sequence(btids, batch_first=True)
            return g_padded,b_padded, gtids_padded, btids_padded

        g_padded = pad_sequence(gcombos, batch_first=True, padding_value=tokenizer.pad_token_id)
        b_padded = pad_sequence(bcombos, batch_first=True, padding_value=tokenizer.pad_token_id)
        gtids_padded = pad_sequence(gtids, batch_first=True, padding_value=tokenizer.pad_token_id)
        btids_padded = pad_sequence(btids, batch_first=True, padding_value=tokenizer.pad_token_id)
        return g_padded,b_padded, gtids_padded, btids_padded
    batch_size = args.batch_size_per_gpu#* max(1, args.n_gpu)


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
