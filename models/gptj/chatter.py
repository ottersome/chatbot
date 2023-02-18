import sys
import torch
from GPTJ8bit import *

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
name = 'EleutherAI/gpt-j-6B'
tokenizer = AutoTokenizer.from_pretrained(name)

if len(sys.argv) > 1: 
    print("Using online model")
    print(sys.argv[1])

    print('Setting Up Tokenizers and (Possibly) PreTrained Models')
    config = AutoConfig.from_pretrained(sys.argv[1], cache_dir='./.my_cache')
    model = AutoModelWithLMHead.from_pretrained(
            sys.argv[1], 
            from_tf=False,
            config=config,
            cache_dir='./.my_cache/')
else:
    #  model = AutoModelWithLMHead.from_pretrained('output/dialoggpt-medium-epoch-20')
    model = GPTJForCausalLM.from_pretrained("output/chkpnt-95", low_cpu_mem_usage=True)


# Let's chat for 5 lines
for step in range(6):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer.encode(input("") + tokenizer.eos_token, return_tensors='pt')
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([chat_history_ids, new_user_input_ids], dim=-1) if step > 0 else new_user_input_ids

    # generated a response while limiting the total chat history to 1000 tokens, 
    chat_history_ids = model.generate(bot_input_ids, min_length=128,max_length=128, do_sample=True)
    
    # pretty print last ouput tokens from bot
    print(tokenizer.decode(chat_history_ids[:, bot_input_ids.shape[-1]:][0], skip_special_tokens=True))
    sys.stdout.flush()


