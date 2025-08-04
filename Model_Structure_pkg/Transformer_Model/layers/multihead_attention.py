from torch import nn
import torch
import math
import torch.nn.functional as F
'''
def ScaleDotProductAttention(nn.Module):
    """
    Scaled Dot-Product Attention
    """
    def __init__(self, dropout=0.1):
        super(ScaleDotProductAttention, self).__init__()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, query, key, value, mask=None):
        """
        Args:
            query: Query tensor of shape (batch_size, num_heads, seq_len_q, d_k)
            key: Key tensor of shape (batch_size, num_heads, seq_len_k, d_k)
            value: Value tensor of shape (batch_size, num_heads, seq_len_v, d_v)
            mask: Optional mask tensor of shape (batch_size, 1, 1, seq_len_k)
        Returns:
            output: Output tensor of shape (batch_size, num_heads, seq_len_q, d_v)
            attention_weights: Attention weights tensor of shape (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        scores = torch.matmul(query, key.transpose(-2, -1)) / torch.sqrt(query.size(-1))
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        output = torch.matmul(attention_weights, value)
        
        return output, attention_weights
'''

class ScaleDotProductAttention(nn.Module):
    """
    compute scale dot product attention

    Query : given sentence that we focused on (decoder)
    Key : every sentence to check relationship with Qeury(encoder)
    Value : every sentence same with Key (encoder)
    """

    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, query, key, value, mask=None, e=1e-12):
        # input is 4 dimension tensor
        # [batch_size, head, length, d_tensor]
        batch_size, head, length, d_tensor = key.size()

        # 1. dot product Query with Key^T to compute similarity

        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))

        # 2. apply masking (opt)
        if mask is not None:
            # Fills elements of self tensor with value where mask is True. And here we set
            # the function to fill pixels with -1e9 where mask is 0
            # The shape of mask must be broadcastable with the shape of the underlying tensor.
            #print("scores.shape:", scores.shape)
            #print("mask.shape:", mask.shape)
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. pass them softmax to make [0, 1] range
        attention_weights = self.softmax(scores)

        # 4. multiply with Value
        output = torch.matmul(attention_weights, value)

        return output, attention_weights



class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, n_head):
        super(MultiHeadAttention, self).__init__()
        self.n_head = n_head
        self.attention = ScaleDotProductAttention()
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_concat = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):

        # 1. dot product with weight matrices
        q, k, v = self.w_q(q), self.w_k(k), self.w_v(v)
        
        # 2. split tensor by number of heads
        q, k, v = self.split(q), self.split(k), self.split(v)

        # 3. do scale dot product to compute similarity
        out, attention = self.attention(q, k, v, mask=mask)

        # 4. concat and pass to linear layer
        out = self.concat(out)
        out = self.w_concat(out)

        # 5. visualize attention map

        return out

    def split(self, tensor):
        """
        split tensor by number of head

        :param tensor: [batch_size, length, d_model]
        :return: [batch_size, head, length, d_tensor]
        """
        batch_size, length, d_model = tensor.size()

        d_tensor = d_model // self.n_head
        tensor = tensor.view(batch_size, length, self.n_head, d_tensor).transpose(1, 2)
        # it is similar with group convolution (split by number of heads)

        return tensor

    def concat(self, tensor):
        """
        inverse function of self.split(tensor : torch.Tensor)

        :param tensor: [batch_size, head, length, d_tensor]
        :return: [batch_size, length, d_model]
        """
        batch_size, head, length, d_tensor = tensor.size()
        d_model = head * d_tensor

        tensor = tensor.transpose(1, 2).contiguous().view(batch_size, length, d_model)
        return tensor