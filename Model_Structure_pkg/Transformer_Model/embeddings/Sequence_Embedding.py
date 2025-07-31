import torch
import torch.nn as nn
import torch.nn.functional as F


### Embedding Layer

class Linear_Embedding(nn.Module):
    """
    Linear embedding layer for input features.
    """

    def __init__(self, input_dim, output_dim):
        ## input dim is the number of the input features
        ## output dim is the number of the model dimensions
        super(Linear_Embedding, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)  # Apply linear transformation to the input tensor
    

class Positional_Embedding(nn.Module):
    """
    Positional embedding layer for input features.
    """

    def __init__(self, d_model, max_len=1000, device=None):
        ## d_model is the number of the model dimensions
        ## max_len is the maximum length of the input sequence
        super(Positional_Embedding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model, device=device)
        self.encoding.requires_grad = False  # Disable gradient computation for positional encoding

        # Initialize positional encoding matrix
        position = torch.arange(0, max_len, dtype=torch.float, device=device).unsqueeze(1)
        # 1D => 2D unsqueeze to represent word position, i.e., make this a column vector

        div_term = torch.arange(0, d_model, 2, dtype=torch.float, device=device)
        div_term = torch.exp(div_term * -(torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
    def forward(self, x):
        """
        Args:
            x: input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x: tensor with positional encoding added
        """
        seq_len = x.size(1)
        x = x + self.encoding[:seq_len, :].unsqueeze(0).detach()
        return x
    
class Sequence_Embedding(nn.Module):
    """
    Sequence embedding layer for input features.
    """

    def __init__(self, input_dim, output_dim, max_len=1000,drop_prob=0.1,device=None): 
        ## input dim is the number of the input features
        ## output dim is the number of the model dimensions
        super(Sequence_Embedding, self).__init__()
        self.linear_embedding = Linear_Embedding(input_dim, output_dim)
        self.positional_embedding = Positional_Embedding(output_dim, max_len,device=device)
        self.dropout = nn.Dropout(drop_prob)  # Dropout layer with a dropout rate. Default: 0.1
    def forward(self, x):
        x = self.linear_embedding(x)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        return x
    
