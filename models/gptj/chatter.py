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
    model = GPTJForCausalLM.from_pretrained(name, low_cpu_mem_usage=True)
    add_adapters(model)
    model.load_state_dict(torch.load('./output/chkpnt-190/model_state_dict.pt'))
    device = torch.device('cuda')
    model.to(device)


print("All good hre , we are about to start")
# Let's chat for 5 lines

for step in range(6):
    # encode the new user input, add the eos_token and return a tensor in Pytorch
    new_user_input_ids = tokenizer(input("") + tokenizer.eos_token, return_tensors='pt')
    new_user_input_ids  = new_user_input_ids.to(device)
    # print(new_user_input_ids)

    # append the new user input tokens to the chat history
    bot_input_ids = torch.cat([bot_input_ids,nth_output, new_user_input_ids['input_ids']], dim=-1) if step > 0 else new_user_input_ids['input_ids']

    # generated a response while limiting the total chat history to 1000 tokens, 
    nth_output = model.generate(input_ids=bot_input_ids, attention_mask=torch.ones_like(bot_input_ids),max_length=128, do_sample=True)
    
    # pretty print last ouput tokens from bot
    print(tokenizer.decode(nth_output[0], skip_special_tokens=True))
    sys.stdout.flush()


