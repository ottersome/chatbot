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
device = torch.device('cuda')
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloomz-7b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloomz-7b1", device_map="auto", load_in_8bit=True)

model.eval()
cur_length = 0
# print('Conversation Starts:')
#
# encode the new user input, add the eos_token and return a tensor in Pytorch
#user_input = input("User: ")
#logging.info("User: " +user_input)
#new_user_input_ids = tokenizer(user_input + tokenizer.eos_token, return_tensors='pt')
new_user_input_ids = tokenizer.encode("What is the capital City of Colombia?", return_tensors="pt").to("cuda")

# append the new user input tokens to the chat history
#bot_input_ids = torch.cat([nth_output, new_user_input_ids['input_ids']], dim=-1) if cur_length > 0 else new_user_input_ids['input_ids']
#cur_length = bot_input_ids.shape[-1]

# generated a response while limiting the total chat history to 1000 tokens, 
#nth_output = model.generate(input_ids=bot_input_ids, do_sample=True)
nth_output = model.generate(new_user_input_ids)

print(tokenizer.decode(nth_output[0]))

print('Done.')
