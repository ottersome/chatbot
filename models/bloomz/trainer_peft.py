import os 
import torch
import logging
import copy
import json
import sys
import wandb
import bitsandbytes as bnb
from datetime import date
from datasets import *
from huggingface_hub import login 
from utils import *
from typing import List, Dict
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from transformers import (
        EvalPrediction,
        AutoModelForCausalLM,
        AutoTokenizer,
        TrainingArguments,
        TrainerState,
        Trainer)
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
from peft import (
        prepare_model_for_int8_training,
        LoraConfig,
        get_peft_model,
        TaskType)
from datetime import datetime



def my_compute_metrics(p: EvalPrediction):

    return {'marco': 1}

class MyTrainer(Trainer):
    def on_train_begin(self, args, state, control, model, tokenizer=None, **kwargs):
        print('We made it here and the globa_step we are setting is: ', checkpoint['epoch'])
        super().on_train_begin(args, state, control, model, tokenizer, **kwargs)
        checkpoint = json.load('trainer_state.json')
        self.state.epoch        = checkpoint['epoch']
        self.state.global_step = checkpoint['global_step']



if __name__ == '__main__':
    #login()
    logger = logging.getLogger(__name__)

    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
                        datefmt="%m/%d/%Y %H:%M:%S",
                        filename='training.log',
                        level=logging.INFO
                        )

    ########################################
    # Load Model
    ########################################
    print('Loading Model...')
    args = parse_args()
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, load_in_8bit=True, device_map="auto", cache_dir = args.cache_dir)
    model.gradient_checkpointing_enable()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir = args.cache_dir)


    print('Model Loaded.')
    tparams,allparams = trainable_params(model)
    print(
            f"Originally: Trainable params: {tparams} || all params: {allparams} || trainable%: {100 * tparams / allparams}"
    )

    print('Configuring Peft...')
    model = prepare_model_for_int8_training(model)
    lora_config = LoraConfig(
            r=16,lora_alpha=32, target_modules=["query_key_value"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            )

    model = get_peft_model(model,lora_config)
    print(
            f"After Peft: Trainable params: {tparams} || all params: {allparams} || trainable%: {100 * tparams / allparams}"
    )
    optimizer,scheduler = (None,None)
    # This is for StepWise
    if args.checkpoint_path != "":
        print('Loading checkpoint from path: '.format(args.checkpoint_path))
        model.load_state_dict(torch.load(args.checkpoint_path+'/pytorch_model.bin'))
        # scheduler = torch.load(args.checkpoint_path+'/scheduler.pt')
        # opt_params = torch.load(args.checkpoint_path+'/optimizer.pt')
        # print(scheduler)
        # optimizer = bnb.optim.Adam8bit(params=model.parameters())
        # optimizer.load_state_dict(opt_params)

        #checkpoint = torch.load(args.checkpoint_path+'/training_state.bin')
    print('Optimizer looks like {}'.format(optimizer))


    ########################################
    # Load Data
    ########################################
    print('Loading Data...')

    train_dataset = BotDataset(tokenizer,args,logger)
    eval_dataset = copy.copy(train_dataset)

    train_dataset.set_mode('train')
    eval_dataset.set_mode('eval')


    ########################################
    # Training
    ########################################
    
    def collator(examples: List[Dict[str,torch.Tensor]]):
        batch = {}
        first = examples[0]
        # Collect Differently
        for k in first.keys():
            batch[k] = [e[k] for e in examples]
        # Now Just Pad it as we know how to 
        padder = 0.0
        if tokenizer._pad_token != None:
            paddier= tokenizer._pad_token
        for k,v in batch.items():
            batch[k] = pad_sequence(v, batch_first=True, padding_value=tokenizer.pad_token_id)

        return batch

    run_name = datetime.strftime(datetime.now(),'%Y_%m_%d-%H-%M_%S_train_run')
    print('Naming run {}'.format(run_name))

    training_args = TrainingArguments(
            "temp",
            evaluation_strategy="steps",
            learning_rate=1e-5,
            gradient_accumulation_steps=2,
            #auto_find_batch_size=True,
            per_device_train_batch_size=4,
            num_train_epochs=1,
            warmup_steps=100,
            logging_dir="runs",
            logging_steps=5,
            save_strategy="steps",
            report_to='wandb',
            eval_steps=12,
            save_steps=3000,
            run_name=run_name,
            save_total_limit=2
            )

    trainer = MyTrainer(
        model=model,
        data_collator=collator,
        args=training_args,
        # optimizers=(optimizer,scheduler),
        train_dataset=train_dataset,
        #compute_metrics=my_compute_metrics,
        eval_dataset=eval_dataset
        )
    # if args.checkpoint_path != "":
    model.config.use_cache = False

    wandb.init(
        project ='parrot',
        config=model.config
            )

    if args.checkpoint_path != "":
        state = TrainerState.load_from_json(args.checkpoint_path+'trainer_state.json')
        trainer.state =state
        last_checkpoint = args.checkpoint_path
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        trainer.train()

    model.push_to_hub("ottersome/bloomy", use_auth_token=True)
    # train_dataloader = DataLoader(
        # train_dataset, 
        # batch_size=None, 
        # num_workers=0  # number of subprocesses to use for data loading
    # )
    # # Train the model and evaluate every 100 steps
    # for step, _ in enumerate(train_dataloader):
        # trainer.train_step(batch)
        # if (step + 1) % 100 == 0:
            # print_evaluation_sample_output(trainer)
            # trainer.evaluate()  # evaluate the model on the evaluation dataset

    # print('Training Finished')
    # TODO Evaluate the model
    # today = date.today()
    # datefmt="%m-%d-%Y_%H-%M-%S"
    # datestr = today.strftime(datefmt)
    # checkpoint_name = f"{args.model_name_or_path}_{datestr}.pt"

