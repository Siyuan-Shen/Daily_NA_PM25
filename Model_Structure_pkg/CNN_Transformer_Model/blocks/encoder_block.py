from torch import nn
from Model_Structure_pkg.CNN_Transformer_Model.layers.multihead_attention import MultiHeadAttention
from Model_Structure_pkg.CNN_Transformer_Model.layers.position_wise_feed_forward import PositionwiseFeedForward


class EncoderBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob=0.1):
        super(EncoderBlock, self).__init__()
        self.attention = MultiHeadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.feed_forward = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.dropout2 = nn.Dropout(p=drop_prob)
        self.layer_norm2 = nn.LayerNorm(d_model)
        

    def forward(self, x, mask=None):
        # Multi-head attention
        attn_output = self.attention(x, x, x, mask=mask)
        x = x + self.dropout1(attn_output)  # Residual connection
        x = self.layer_norm1(x)  # Layer normalization

        # Feed-forward network
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)  # Residual connection
        x = self.layer_norm2(x)  # Layer normalization

        return x