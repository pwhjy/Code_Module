import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    """
    if mask is not None:
        mask = mask.float()
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # 这里(mask+1e-45)是因为当整个vector都被mask的时候
        # 结果就会变成nan,所以通过加一个很小的值来防止出现这种
        # 情况,1e-45是能够取到的最小的值,1e-46就太小了
        print(mask.shape)
        vector = vector + (mask + 1e-45).log()
    # log_softmax在softmax的基础上加了一次log
    return F.log_softmax(vector, dim=dim)

def masked_max(vector: torch.Tensor,
			   mask: torch.Tensor,
			   dim: int,
			   keepdim: bool = False,
			   min_val: float = -1e7):
    """
    """
    one_minus_mask = (1.0 - mask).byte()
    replaced_vector = vector.masked_fill(one_minus_mask, min_val)
    max_value, max_index = replaced_vector.max(dim=dim, keepdim=keepdim)
    return max_value, max_index

class EncoderDNN(nn.Module):
    """
    Encoder负责编码所有输入,包含static,dynamic,每个输入对应使用一个Encoder网络.
    """
    def __init__(self) -> None:
        super(EncoderDNN, self).__init__()

class EncoderRNN(nn.Module):
    """
    Encoder负责编码所有输入,包含static,dynamic,每个输入对应使用一个Encoder网络.
    """
    def __init__(self, embedding_dim, hidden_size, num_layers=1, batch_first=True, bidirectional=True):
        super(EncoderRNN, self).__init__()
        self.batch_first = batch_first
        self.rnn = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
						   batch_first=batch_first, bidirectional=bidirectional)

    def forward(self, embedded_inputs, input_lengths):
        # Pack padded batch of sequences for RNN module
        packed = nn.utils.rnn.pack_padded_sequence(embedded_inputs, input_lengths, batch_first=self.batch_first)
        # Forward pass through RNN
        outputs, hidden = self.rnn(packed)
        # Unpack padding
        outputs, _ = nn.utils.rnn.pad_packed_sequence(outputs, batch_first=self.batch_first)
        # Return output and final hidden state
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, hidden_size) -> None:
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.W1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.vt = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, decoder_state, encoder_outputs, mask):
        """
        -----------------
        eq1: uij=vt*tanh(W1*ej+W2*di)
        eq2: p(Ci|C1,...,Ci-1,P)=softmax(ui)
        -----------------
        input:
        decoder_state:
        encoder_outputs:
        mask:
        -----------------
        output:

        """
        #(batch_size, max_seq_len, hidden_size)
        encoder_transform = self.W1(encoder_outputs)
        #(batch_size, 1, hidden_size)
        decoder_transform = self.W2(decoder_state).unsqueeze(1)
        # (batch_size, max_seq_len, 1) => (batch_size, max_seq_len)
        ui = self.vt(torch.tanh(encoder_transform + decoder_transform)).squeeze(-1)
        log_score = masked_log_softmax(ui, mask, dim=-1)
        return log_score

class PtrNetwork(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_size, bidirectional=True, batch_first=True) -> None:
        super(PtrNetwork, self).__init__()

        # Embedding dimension
        self.embedding_dim = embedding_dim
        # (Decoder) hidden size
        self.hidden_size = hidden_size
        # Bidirectional Encoder
        self.bidirectional = bidirectional
        self.num_direction = 2 if bidirectional else 1
        self.num_layers = 1
        self.batch_first = batch_first

        # We use an embedding layer for more complicate application usages later, e.g. word sequences.
        self.embedding = nn.Linear(in_features=input_dim, out_features=embedding_dim, bias=False)
        self.encoder = EncoderRNN(embedding_dim=embedding_dim, hidden_size=hidden_size, num_layers=self.num_layers,
                                  bidirectional=bidirectional, batch_first=batch_first)

        self.decoding_rnn = nn.LSTMCell(input_size=hidden_size, hidden_size=hidden_size)
        self.attention = Attention(hidden_size=hidden_size)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(self, input, input_length):
        if self.batch_first:
            batch_size = input.size(0)
            max_seq_len = input.size(1)
        else:
            batch_size = input.size(1)
            max_seq_len = input.size(0)
        
        # Embedding
        embedded = self.embedding(input)
        # encoder_output => (batch_size, max_seq_len, hidden_size) if batch_first else (max_seq_len, batch_size, hidden_size)
		# hidden_size is usually set same as embedding size
		# encoder_hidden => (num_layers * num_directions, batch_size, hidden_size) for each of h_n and c_n
        encoder_outputs, encoder_hidden = self.encoder(embedded, input_length)
        if self.bidirectional:
            encoder_outputs = encoder_outputs[:, :, :self.hidden_size] + encoder_outputs[:, :, self.hidden_size]

        encoder_h_n, encoder_c_n = encoder_hidden
        encoder_h_n = encoder_h_n.view(self.num_layers, self.num_direction, batch_size, self.hidden_size)
        encoder_c_n = encoder_c_n.view(self.num_layers, self.num_direction, batch_size, self.hidden_size)

        decoder_input = encoder_outputs.new_zeros(torch.Size((batch_size, self.hidden_size)))
        decoder_hidden = (encoder_h_n[-1, 0, :, :].squeeze(), encoder_c_n[-1, 0, :, :].squeeze())

        range_tensor = torch.arange(max_seq_len, device=input_length.device, dtype=input_length.dtype).expand(batch_size, max_seq_len, max_seq_len)
        each_len_tensor = input_length.view(-1, 1, 1).expand(batch_size, max_seq_len, max_seq_len)
        
        row_mask_tensor = (range_tensor < each_len_tensor)
        col_mask_tensor = row_mask_tensor.transpose(1, 2)
        mask_tensor = row_mask_tensor * col_mask_tensor
        
        pointer_log_scores = []
        pointer_argmaxs = []

        for i in range(max_seq_len):
            sub_mask = mask_tensor[:, i, :].float()

            # h, c: (batch_size, hidden_size)
            h_i, c_i = self.decoding_rnn(decoder_input, decoder_hidden)

            # next hidden
            decoder_hidden = (h_i, c_i)

            # Get a pointer distribution over the encoder outputs using attention
            # (batch_size, max_seq_len)
            log_pointer_score = self.attention(h_i, encoder_outputs, sub_mask)
            pointer_log_scores.append(log_pointer_score)

            # Get the indices of maximum pointer
            _, masked_argmax = masked_max(log_pointer_score, sub_mask, dim=1, keepdim=True)

            pointer_argmaxs.append(masked_argmax)
            index_tensor = masked_argmax.unsqueeze(-1).expand(batch_size, 1, self.hidden_size)

            # (batch_size, hidden_size)
            decoder_input = torch.gather(encoder_outputs, dim=1, index=index_tensor).squeeze(1)

        pointer_log_scores = torch.stack(pointer_log_scores, 1)
        pointer_argmaxs = torch.cat(pointer_argmaxs, 1)
    
        return pointer_log_scores, pointer_argmaxs, mask_tensor