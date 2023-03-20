import sys
import torch
import logging
from datetime import datetime
from GPTJ8bit import *
from pathlib import Path

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
#pipe = pipeline(model='EleutherAI/gpt-j-6B',model_kwargs={'device_map':"auto","load_in_8_bits":True})
name = 'hivemind/gpt-j-6B-8bit'
tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-j-6B')


if len(sys.argv) > 1: 
    #  model = AutoModelWithLMHead.from_pretrained('output/dialoggpt-medium-epoch-20')
    model = GPTJForCausalLM.from_pretrained(name, low_cpu_mem_usage=True)
    add_adapters(model)
    model.load_state_dict(torch.load(sys.argv[1]+'/model_state_dict.pt'))
    device = torch.device('cuda')
    model.to(device)
else:
    # print("Using online model")
    # print('Setting Up Tokenizers and (Possibly) PreTrained Models')
    config = AutoConfig.from_pretrained(name, cache_dir='./.my_cache')
    model = AutoModelWithLMHead.from_pretrained(
            name, 
            from_tf=False,
            config=config,
            cache_dir='./.my_cache/')


# Let's chat for 5 lines
stem = Path(sys.argv[1]).stem
now = datetime.now()
logging.basicConfig(filename=now.strftime('./chats/ckpnt_{}_%Y-%m-%d_%H:%M:%S'.format(stem)),
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

logging.info("Starting the debugging")
cur_length = 0
# print('Conversation Starts:')
while True:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    user_input = input("")
    logging.info("User: " +user_input)
    user_input = tokenizer.eos_token.join(user_input.split('|')) 

    logging.debug("Formatting input to  :{}".format(user_input))
    
    new_user_input_ids = tokenizer(user_input, return_tensors='pt')
    new_user_input_ids  = new_user_input_ids.to(device)
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    # Create the mask.

    # generated a response while limiting the total chat history to 1000 tokens, 
    nth_output = model.generate(input_ids=new_user_input_ids['input_ids'],max_length=2048, do_sample=True)
    cur_length = new_user_input_ids['input_ids'].shape[-1]
    
    decoded_output = tokenizer.decode(nth_output[0][cur_length:], max_length=cur_length+1000, skip_special_tokens=True, temperature=0.9, pad_token_id =0)
    logging.info('Bot: '+decoded_output)

    # print("{}".format(len(nth_output[0]),decoded_output))
    print("{}".format(decoded_output))
    sys.stdout.flush()


