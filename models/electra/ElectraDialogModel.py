from torch import nn
class DialogModel(nn.Module):
    def __init__(self,base_model, hidden_size, num_classes):
        self.pretrained_model = model
        # Use Dropout(For avoiding overfitting im guessing)
        self.dropout = nn.Dropout(0.1)
         

