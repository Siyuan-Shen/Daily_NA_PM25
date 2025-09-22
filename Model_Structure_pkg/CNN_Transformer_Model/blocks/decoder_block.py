import torch.nn as nn

from Model_Structure_pkg.CNN_Transformer_Model.layers.multihead_attention import MultiHeadAttention
from Model_Structure_pkg.CNN_Transformer_Model.layers.position_wise_feed_forward import PositionwiseFeedForward


class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, ffn_hidden, drop_prob=0.1):
        super(DecoderBlock, self).__init__()
        self.attention1 = MultiHeadAttention(d_model, n_head)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=drop_prob)

        self.enc_dec_attention = MultiHeadAttention(d_model, n_head)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(p=drop_prob)

        self.feed_forward = PositionwiseFeedForward(d_model, ffn_hidden, drop_prob)
        self.dropout3 = nn.Dropout(p=drop_prob)
        self.layer_norm3 = nn.LayerNorm(d_model)
    def forward(self, dec_input, enc_output, target_mask=None, src_mask=None):
        # The target mask is used to prevent the decoder from attending to future tokens
        # in the sequence during training (causal masking).
        # The enc_dec_mask is used to mask the invalid positions in the encoder output.
        attn_output = self.attention1(dec_input, dec_input, dec_input, mask=target_mask)
        dec_input = dec_input + self.dropout1(attn_output)  # Residual connection
        dec_input = self.layer_norm1(dec_input)

        # Encoder-decoder attention
        if enc_output is not None:
            enc_dec_output = self.enc_dec_attention( dec_input,  enc_output, enc_output, mask=src_mask)
            dec_input = dec_input + self.dropout2(enc_dec_output)
            dec_input = self.layer_norm2(dec_input)
        # Feed-forward network
        ff_output = self.feed_forward(dec_input)
        dec_input = dec_input + self.dropout3(ff_output)
        dec_input = self.layer_norm3(dec_input)
        return dec_input

