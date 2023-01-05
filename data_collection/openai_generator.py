import os
import openai
import argparse
import csv

openai.api_key = os.getenv("OPENAI_API_KEY")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_dialog_id",
        dest='init_diag_id',
        default=0,
        type=int)
    parser.add_argument("--init_utterance_id",
        dest='init_utterance_id',
        default=0,
        type=int)
    parser.add_argument("--prompt",
        dest='prompt',
        default="What follows is a creative conversation between two friends over the phone. These characters are supporting, inquisitive and good friends. Character 1 is sharing their day. Character 2 is lending an ear and providing good advice. The conversation is around 550 words in length. Responses between chracters are formatted as csv \"response_id, response_text\". 2 Example rows: '1|\"Hey How are you?\"\n2|\"Im fine\"\n1|\"Thats good to hear\"' \n[insert]",
        type=str)
    parser.add_argument("--temperature",
        dest='temperature',
        default=0.5,
        type=int)
    parser.add_argument("--num_conversations",
        dest='num_convos',
        required=True,
        type=int)
    parser.add_argument("--max_convo_size",
        dest='max_convo_size',
        default=2048,
        type=int)
    parser.add_argument("--model",
        dest='model',
        default="text-davinci-003",
        type=str)
    parser.add_argument("--output_file_dump",
        dest='output_file_dump',
        default="output_convos.csv",
        type=str)

    return parser.parse_args()

if __name__=="__main__":

    args = parse_args()
    actual_convos = 0
    output_file = open(args.output_file_dump,"w")
    utt_id = args.init_utterance_id
    diag_id = args.init_diag_id
    print("Prompt being used is:\n", args.prompt)
    for i in range(args.num_convos):
        # Form the request
        response = openai.Completion.create(
            model=args.model,
            #prompt="I want you to act like a friend. I want you to respond and answer like friend using the tone, manner and vocabulary a casual friend would use. Do not write any explanations. Only answer like a friend. You responses must sound supportive, helpful, inquisitive and informative. I will act as a friend who's thinking about getting a new pet. \". Keep your responses brief(1-2 sentences) unless absolutely necessary. My first utterance is \"I think I want to get a pet. I've been feeling really lonely lately and I think a small furry companion might help me.",
            #prompt="Generate 3 different prompts about conversations people may have over the phone. Then, take those 5 prompts and generate conversations of no longer than 500 words each. The output of the entire thing should be in a json format, readily available to be used as dataset.",
            prompt=args.prompt,
            temperature=args.temperature,
            max_tokens=args.max_convo_size,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0
            )
        # Work with response
        if response['choices'][0]['finish_reason'] != "stop":
            continue
        # Otherwise parse the conversation
        raw_convo = response['choices'][0]['text'].strip('\n')
        csv_convo = csv.reader(raw_convo,delimiter=',')

        # This is my custom part
        for row in raw_convo.split('\n'):
            print(row)
            convo_split = row.split('|')
            output_file.write("{}|{}|||{}|{}\n".format(utt_id,diag_id, convo_split[0], convo_split[1]))
            utt_id+=1
        
        diag_id += 1
        actual_convos += 1 

    print("Final amount of conversation is %d/%d".format(actual_convos, args.num_convos))

print(response)
