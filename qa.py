import torch
from transformers import BertForQuestionAnswering
from transformers import BertTokenizer

#Model
model = BertForQuestionAnswering.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

#Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-large-uncased-whole-word-masking-finetuned-squad')

print("Prompting...")
context = input('Please give me some context\n')
question = input('What is your quesiton\n')

# question = '''Why was the student group called "the Methodists?"'''

# context = ''' The movement which would become The United Methodist Church began in the mid-18th century within the Church of England.
            # A small group of students, including John Wesley, Charles Wesley and George Whitefield, met on the Oxford University campus.
            # They focused on Bible study, methodical study of scripture and living a holy life.
            # Other students mocked them, saying they were the "Holy Club" and "the Methodists", being methodical and exceptionally detailed in their Bible study, opinions and disciplined lifestyle.
            # Eventually, the so-called Methodists started individual societies or classes for members of the Church of England who wanted to live a more religious life. '''

encoding = tokenizer.encode_plus(text=question,text_pair=context,add_special=True)

inputs= encoding['input_ids']
sentence_embedding = encoding['token_type_ids']
tokens = tokenizer.convert_ids_to_tokens(inputs)



print("Looking for your answer")
results = model(
        input_ids=torch.tensor([inputs]),
        token_type_ids=torch.tensor([sentence_embedding])
        )

start_index = torch.argmax(results['start_logits'])
end_index = torch.argmax(results['end_logits'])

answer = ' '.join(tokens[start_index:end_index+1])
corrected_answer = ''
for word in answer.split():
    if word[0:2] == "##":
        corrected_answer+=word[2:]
    else:
        corrected_answer +=' '+word
print('Answer is:\n',corrected_answer)



