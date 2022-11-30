from transformers import AutoModelWithLMHead, AutoTokenizer
import torch
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Chatbot')
    parser.add_argument('--convlength',
                        dest='conv_length',
                        default=1000,
                        type=int,
                        help='Length of conversation')

    return parser.parse_args()


if __name__ == '__main__':
    # Do the Parseing
    args = parse_arguments()

    # Get Tokenizer and Model
    tokenizer = AutoTokenizer.from_pretrained("microsoft/DialoGPT-medium")
    model = AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-medium")

    # Get Some Context
    #context = tokenizer.encode()

    # Create n step conversaiton 
    print("Size of the encoded vectors is : "+ ' '.join(str(tokenizer.encode(tokenizer.eos_token))))
    chat_history = torch.empty((1,1))
    for step in range(args.conv_length):
        latest_input_tokens = tokenizer.encode(input("Your Question >>")+tokenizer.eos_token, return_tensors='pt')
        inputs_for_bot = torch.cat([chat_history, latest_input_tokens],dim=-1) if step > 0 else latest_input_tokens

        chat_history = model.generate(
                inputs_for_bot,
                max_length=1000,
                pad_token_id=tokenizer.eos_token_id,
                top_p=0.95,
                top_k=50,
                #temperature=0.6,
                repetition_penalty=1.3)
        print(chat_history)
        print("Bot {}:".format(tokenizer.decode(chat_history[:,inputs_for_bot.shape[-1]:][0],skip_special_tokens=True)))






