import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, hidden_size, embedding_matrix, padding_idx=0, freeze_embeddings=True, word_dropout=0.):
        """
        hidden_size: size of the recurrent state
        embedding_matrix: word vectors
        padding_idx: id of the pad token
        freeze_embeddings: treat embeddings as frozen
        """
        super().__init__()
        vocab_size, embedding_dim = embedding_matrix.shape

        # Pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix,
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )
        self.hidden_size = hidden_size
        self.word_dropout = word_dropout
        
        # LSTM
        self.lstm = nn.LSTM(
            input_size=embedding_dim, 
            hidden_size=hidden_size,
            num_layers=1, # for sure tune (1,2)
            batch_first=True,  # important, in NLP we treat the first axis of the tensor as the batch dimension, and the second as the sequence dimension
            )

    @property
    def enc_dim(self):
        return self.hidden_size

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):
        """
        Encode each sequence in the batch into a single H-dimensional vector.
        
        x: a batch of sequences of token ids
        """
        # [...,max_seq_len, emb_dim]
        embedded = self.embedding(x)   
        valid = x.ne(self.embedding.padding_idx)

        # Word dropout is a regularisation technique for recurrent architectures
        # If you are curious why we have to drop words in order to regularise RNNs, 
        # here's the paper that figured it out: https://proceedings.neurips.cc/paper_files/paper/6241-a-theoretically-grounded-application-of-dropout-in-recurrent-neural-networks.pdf
        # In a nutshell, if we apply dropout before linear layers inside the RNN cell
        # as sequences grow long, information is completely destroyed.
        # That paper builds on a previous paper that interpreted dropout as a specific kind of approximate inference (for a specific kind of approximate Bayesian model)
        # and then establishes that to achieve the same in a recurrent architecture, one can drop words as we do here
        # (By dropping a word, we mean 0-out its embedding)
        if self.training and self.word_dropout > 0:
            # Independently zero each non-padding token with probability word_dropout.
            # [..., max_seq_len]
            drop_mask = (
                torch.rand(x.shape, device=embedded.device) < self.word_dropout # at a certain pre-specified rate
            ) & valid
            # we 0-out the embedding of a valid (not pad) position
            embedded = embedded.masked_fill(drop_mask.unsqueeze(-1), 0.0)
                
        # [..., max_seq_len] this is necessary for packing
        lengths = torch.sum(valid, dim=-1)
        # for efficiency, we let torch do the for loop, and for that we need to 'pack' our sentences
        # using a torch helper
        packed = pack_padded_sequence(embedded, lengths.cpu(), batch_first=True, enforce_sorted=False) # sorting leads to even more efficient processing, but it is cumbersome and we skip it
        output, (hidden, cell) = self.lstm(packed)
        final_hidden = hidden[-1]
        return final_hidden 
    
