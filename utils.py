import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence 
from torch.nn.utils.rnn import pad_sequence 
import time
from tqdm.auto import tqdm

from zipfile import ZipFile
from io import TextIOWrapper
from pathlib import Path
import gzip
import nltk
import re
import random
import string
from pathlib import Path
from urllib.request import urlopen, urlretrieve
import shutil


def cos_similarity(u, v):
    denominator = np.linalg.norm(u) * np.linalg.norm(v)
    return float(np.dot(u, v) / denominator)


def download_pretrained(option):
    """
    You can use this to download GloVe (or similar) model using wget. 
    It's fine to download it through other means, but don't unzip the file (the rest of the notebook can 
    handle the file in its zipped format).

    This function returns the path which you can use for the loader class that follows.

    The function will skip the download if it detects the file in disk. If the file in disk is corrupted, you will have to delete the file first.
    """
    if option == 'glove':
        destination = "glove.2024.wikigiga.50d.zip"
        dims = 50
        skip_first = False
    elif option == 'fasttext':
        destination = "wiki-news-300d-1M.vec.zip"
        dims = 300
        skip_first = True
    else:
        raise ValueError(f"Unkonwn option: {option}")
        
    if not Path(destination).is_file():
        if option == 'glove':
            urlretrieve("https://nlp.stanford.edu/data/wordvecs/glove.2024.wikigiga.50d.zip", destination)
        elif option == 'fasttext':
            urlretrieve("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M.vec.zip", destination)    

    return destination, dims, skip_first


def collate_sequences(sequences, padding_value=0):
    """
    Pad variable-length token sequences into a tensor.

    Returns:
        Tensor of shape (batch_size, max_seq_length)
    """
    text = pad_sequence(
        [torch.tensor(seq, dtype=torch.long) for seq in sequences],
        batch_first=True,
        padding_value=padding_value
    )
    return text


class Vocabulary:
    """
    This is a helper class that wraps around a list of symbols/words/tokens and provides 
    functions to map from str to int.
    """

    def __init__(self, init: list, default: str, frozen=False):
        if default not in init:
            raise ValueError("The default token must be in the init list")        
        self.words = list(init)
        self._str2int = {w: i for i, w in enumerate(self.words)}
        self._default_word = default
        self._default_index = self._str2int[self._default_word]        
        self._frozen = frozen

    @property
    def default_word(self):
        return self._default_word

    @property
    def default_index(self):
        return self._default_index
        
    def freeze(self):
        """After this operation add(.) no longer expands the vocabulary"""
        self._frozen = True

    def add(self, word: str) -> int:
        """
        Add a word to the vocabulary and return its id (existing words are not added again).
        If the vocabulary is frozen this adds nothing and returns the id of the default.
        """        
        if self._frozen:
            return self.index(word)
        
        i = self._str2int.get(word, -1)
        if i < 0:
            i = len(self.words)
            self.words.append(word)
            self._str2int[word] = i
        return i
            
    def index(self, word: str) -> int:
        """Convert word to 0-based integer (0-based because Python lists are 0-based)"""
        return self._str2int.get(word, self.default_index)

    def word(self, index: int) -> str:
        """Convert an 0-based integer to a word"""
        if index >= len(self):  # if the index is out of bounds, we return the default str
            return self.default_word
        else:
            return self.words[index]  

    def __contains__(self, word: str):
        return word in self._str2int
        
    def __len__(self):
        """Size of vocabulary"""
        return len(self.words)

    def __iter__(self):
        """Iterate over words"""
        return iter(self.words)

    def items(self):
        """Iterates over pairs of the kind (index, word) for all words in the vocabulary"""
        return enumerate(self.words)

    def __str__(self):
        return "\n".join(f"{i} {w}" for i, w in self.items())


class Tokenizer:
    def tokens(self, text: str) -> list:
        """Tokenize as a list of string tokens"""
        raise NotImplementedError("Implement me in subclass")
    def ids(self, text: str) -> list:
        """Tokenize as a list of token ids"""
        raise NotImplementedError("Implement me in subclass")


class BasicTokenizer(Tokenizer):
    """
    Tokenize text (str) using the call method of the object.
    
    Here are the  steps:
    1. lowercase (optional)
    2. remove punctuation (optional)
    3. NLT-tokenize 
    4. remove stopwords (optional)

    Different languages are only supported to the extent that NLTK supports them. 
    If you need something more complex, you need to write your own tokenizer.
    """

    def __init__(self, lowercase: bool, remove_punct: bool, remove_stopwords: bool, language='english', vocab: Vocabulary=None):
        """        
        lowercase: lowercase the text before tokenization
        remove_punct: remove punctuation 
        remove_stopwords: use NLKT to remove stopwords
        language: tells NLTK what language-specific rules to use
        vocab: a Vocabulary object that maintains a correspondence between string symbols and unique ids
         including a 'default' symbol for unkonwn words.
         if None, the basic tokenizer will not be able to implement the `ids` method
        """
        self.vocab = vocab
        self.language = language
        self.lowercase = lowercase
        self.remove_punct = remove_punct
        self.remove_stopwords = remove_stopwords
        if remove_stopwords:
            self.stopwords = set(nltk.corpus.stopwords.words(language))    

    def tokens(self, text) -> 'list[str]':
        """Chops text into a list of words/tokens (each a string)"""
        if self.lowercase:
            text = text.lower()
        if self.remove_punct:
            # sub space for punctuation
            text = re.sub(r'[^\w\s]', ' ', text)        
        # basic NLT tokenization
        tokens = nltk.tokenize.word_tokenize(text, language=self.language)
        if self.remove_stopwords:
            # filter stopwords
            tokens = [token for token in tokens if token not in self.stopwords]
        return tokens

    def ids(self, text) -> 'list[int]':
        """
        Chops text into a list of words/tokens ids.
        Unknown tokens get the id of the default symbol in the vocabulary.
        """
        if self.vocab is None:
            raise ValueError("I need a vocabulary")
        return [self.vocab.index(t) for t in self.tokens(text)]        
        

def read_lines_from_file(path, encoding="utf-8"):
    if path.endswith('.txt'):
        with open(path, 'r', encoding=encoding) as f:
            for line in f:
                yield line.strip()
    elif path.endswith('.gz'):
        with gzip.open(path, 'rt', encoding=encoding) as f:
            for line in f:
                yield line.strip()
    elif path.endswith('.zip'):
        with ZipFile(path) as archive:
            members = archive.namelist()

            if len(members) != 1:
                raise ValueError(
                    f"Expected exactly one .txt file in {path}, "
                    f"found {len(members)}: {members}"
                )

            with archive.open(members[0]) as f:
                with TextIOWrapper(f, encoding=encoding) as text:
                    for line in text:
                        yield line.strip()        
    else:
        raise ValueError(f"Unknown format: {path}")

        
class PretrainedEmbeddings:
    """
    Load an embedding matrix (stored in one of the supported files, see read_lines_from_file)
     and gives structured access to it.     
    """

    def __init__(self, path, num_dims, unk_token="<unk>", pad_token="<pad>", encoding="utf-8", skip_first_line=False):
        """
        path: text/gz/zip file with embeddings
        num_dims: dimensionality of embeddings
        unk_token: special token reserve for words not in the table, we use the average of all embeddings as the unk embedding
        pad_token: special token used for batching sequences of variable length, we use a vector of zeros
        encoding: encoding of the embeddings file
        skip_first_line: some embedding files have metadata in the first line
        """
        vectors = [            
            np.zeros(num_dims, dtype=np.float32),  # pad
            np.zeros(num_dims, dtype=np.float32) # unk placeholder
        ]        
        self.vocab = Vocabulary(init=[pad_token, unk_token], default=unk_token)
        self.num_dims = num_dims        
        self.unk_token = unk_token
        self.pad_token = pad_token 
        
        for line_num, line in enumerate(read_lines_from_file(path, encoding=encoding)):            
            if skip_first_line and line_num == 0:
                continue
            parts = line.split()
            word = parts[0].lower() # as we don't know much about their preprocessing, let's lowercase words here
            values = parts[1:]            
            if len(values) != num_dims: # avoids some known problems with these files
                continue
            vector = np.array(values).astype(np.float32)
            if word in self.vocab:
                continue  # better skip clashes, if any
            self.vocab.add(word)
            vectors.append(vector) 
            
        # [num_words, num_dims]    
        vectors = np.stack(vectors, axis=0)
        # unk embedding is the mean of all others after pad (that's the "center" of the space)
        vectors[self.vocab.default_index] = np.mean(vectors[2:, :], 0)
        self.vectors = vectors
        # we freeze the vocab to avoid having it out of sync with the vectors
        self.vocab.freeze()
        assert self.vectors.shape[0] == len(self.vocab), "Mismatch between vocab and vectors"
        
    def __contains__(self, word: str):
        """Test whether or not a word (or its lowercased version) is in the vocabulary"""
        return word.lower() in self.vocab

    def index(self, word: str):        
        """Return the id of a word (or of its lowercased version) or 0 (by definition, id of unk_token)"""
        return self.vocab.index(word.lower())

    def word(self, i: int):    
        """Return the word associated with an id"""
        return self.vocab.word(i)        

    def __getitem__(self, i: int):
        """Return self.vectors[i]"""
        return self.vectors[i]  # this will raise an exceptino if i is out of bounds
        
    def embed_word(self, word: str):
        """Return the vector representation of the word (or its lowercased version, or unk)"""
        i = self.index(word)
        return self.vectors[i]
    
    def embed_sequence(self, tokens: list):        
        """Return a list of embeddings, one for each token in the input list"""
        return [self.embed_word(t) for t in tokens]

class EmbedAndPoolEncoder(nn.Module):
    """
    This encoder embeds a sequence of token ids and then pools the result along sequence length using
    sum or mean.
    """
    def __init__(self, embedding_matrix, average_pool=True, padding_idx=0, freeze_embeddings=True):
        """
        embedding_matrix: a torch tensor of shape (vocab_size, emb_dim)
        average_pool: use True for mean and False for sum
        padding_idx: the id of the pad token (this is for batched computation)
            suppose we are batching three sequences [1, 2, 3], [6, 2] and [2, 1, 4, 5]
            as they differ in length, we use the following trick to batch them into a single tensor
            [
                [1, 2, 3, 0],
                [6, 2, 0, 0],
                [2, 1, 4, 5]
            ]
            that is we extend the shorter sequences with 0s until they match the longest sequence
        freeze_embeddings: tells torch that we are not updating the parameters in the embedding matrix
        """
        super().__init__()
        self.average_pool = average_pool
        # Pretrained embeddings
        self.embedding = nn.Embedding.from_pretrained(
            embedding_matrix, 
            freeze=freeze_embeddings,
            padding_idx=padding_idx
        )       

    @property
    def enc_dim(self):
        return self.embedding.embedding_dim

    @property
    def pad_idx(self):
        return self.embedding.padding_idx

    def forward(self, x):     
        """
        Encodes each sequence in the batch as a single vector of size emb_dim.
        
        x: token ids with shape (batch_size, max_seq_length, emb_dim)        
        """
        # [..., max_seq_length, emb_dim]
        e = self.embedding(x)  
        # Normalization for pooling
        if self.average_pool:
            # Total non-padding tokens (min of 1 to avoid problems when averaging the sum of embs)
            # [..., max_seq_length, 1]  # keepdim=True keeps the last axis, this is needed for division later on
            den = torch.sum(x.ne(self.embedding.padding_idx), dim=-1, keepdim=True).clamp_min(1)
        else:  # sum pooling
            den = 1
        # [..., emb_dim]
        h = torch.sum(e, -2) / den  # see that we reduce-sum the sequence dimension, not the representation dimension
        return h  