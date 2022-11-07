from torch import nn
class AutoRegressiveDialogModel(nn.Module):
    
    # Base_model is discriminator
    def __init__(self,base_model, hidden_size, num_classes):
        self.discriminator = base_model
        # Use Dropout(For avoiding overfitting im guessing)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size,num_classes)

    # TODO: maybe special inistialization of weights ?
    def forward(self, inpt_ids, attention_mask, token_type_ids):
        # Remember Discriminator itself takes input ids
        x = self.discriminator(input_ids=input_ids, attention_mask = attention_mask, token_type_ids = token_type_ids)[0]
        # Dropout for generalization
        return self.classifier(self.dropout())

    def train():
        optimizer = torch.optim.Adamax(self.discrimnator.parameters)

         

