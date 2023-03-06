import sys
import torch
from GPTJ8bit import *

from torch.nn.utils.rnn import pad_sequence
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
    print("Using online model")

    print('Setting Up Tokenizers and (Possibly) PreTrained Models')
    config = AutoConfig.from_pretrained(name, cache_dir='./.my_cache')
    model = AutoModelWithLMHead.from_pretrained(
            name, 
            from_tf=False,
            config=config,
            cache_dir='./.my_cache/')


print("All good hre , we are about to start")
# Let's chat for 5 lines

cur_length = 0
while True:
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer(input("") + tokenizer.eos_token, return_tensors='pt')
    new_user_input_ids  = new_user_input_ids.to(device)
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([nth_output, new_user_input_ids['input_ids']], dim=-1) if cur_length > 0 else new_user_input_ids['input_ids']
    cur_length = bot_input_ids.shape[-1]

    # generated a response while limiting the total chat history to 1000 tokens, 
    nth_output = model.generate(input_ids=bot_input_ids,max_length=2048, do_sample=True)
    
    # pretty print last ouput tokens from bot
    # print("botinput ids: ", bot_input_ids)
    # print("Nth output:\n\t", nth_output)

    print(tokenizer.decode(nth_output[0][cur_length:], max_length=cur_length+32, skip_special_tokens=True))
    print("Cur Lenght is " , len(nth_output[0]))
    sys.stdout.flush()


