import sys
import torch
import logging
from datetime import datetime
from GPTJ8bit import *

from torch.nn.utils.rnn import pad_sequence
from transformers import utils
from transformers import (
    MODEL_WITH_LM_HEAD_MAPPING,
    WEIGHTS_NAME,
    AdamW,
    AutoConfig,
    pipeline,
    AutoModelWithLMHead,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    get_linear_schedule_with_warmup,
)
from peft import (
        prepare_model_for_int8_training,
        LoraConfig,
        get_peft_model,
        TaskType)
#pipe = pipeline(model='EleutherAI/gpt-j-6B',model_kwargs={'device_map':"auto","load_in_8_bits":True})
name = 'bigscience/bloomz-7b1'
tokenizer = AutoTokenizer.from_pretrained(name)

if len(sys.argv) > 1: 
    device = torch.device('cuda')
    model = AutoModelForCausalLM.from_pretrained(name, load_in_8bit=True, device_map='auto')
    # model = prepare_model_for_int8_training(model)
    # lora_config = LoraConfig(
            # r=16,lora_alpha=32, target_modules=["query_key_value"], lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
            # )
    # model = get_peft_model(model,lora_config)

    # model.load_state_dict(torch.load(sys.argv[1]+'/pytorch_model.bin'))

else:
    # print("Using online model")
    # print('Setting Up Tokenizers and (Possibly) PreTrained Models')
    config = AutoConfig.from_pretrained(name, cache_dir='./.my_cache')
    model = AutoModelWithLMHead.from_pretrained(
            name, 
            from_tf=False,
            config=config,
            cache_dir='./.my_cache/')


model.eval()
cur_length = 0
# print('Conversation Starts:')
while True:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    user_input = input("")
    logging.info("User: " +user_input)
    new_user_input_ids = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt')
    #new_user_input_ids = tokenizer(user_input , return_tensors='pt')
    new_user_input_ids  = new_user_input_ids.to(device)
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([nth_output, new_user_input_ids['input_ids']], dim=-1) if cur_length > 0 else new_user_input_ids['input_ids']
    cur_length = bot_input_ids.shape[-1]

    # generated a response while limiting the total chat history to 1000 tokens, 
    nth_output = model.generate(input_ids=bot_input_ids,max_length=2048, do_sample=True)
    
    decoded_output = tokenizer.decode(nth_output[0][cur_length:], max_length=cur_length+1000, skip_special_tokens=True, pad_token_id =0)
    logging.info('Bot: '+decoded_output)
    # pretty print last ouput tokens from bot
    # print("botinput ids: ", bot_input_ids)
    # print("Nth output:\n\t", nth_output)

    # print("{}".format(len(nth_output[0]),decoded_output))
    print("{}".format(decoded_output))
    sys.stdout.flush()


