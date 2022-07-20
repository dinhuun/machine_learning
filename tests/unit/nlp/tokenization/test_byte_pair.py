from machine_learning.nlp.strings import byte_pair_str
from machine_learning.nlp.tokenization.byte_pair import (
    TOKENIZER,
    count_pair_freqs,
    expand_vocab,
    init_vocab,
    tokenize,
)
from machine_learning.nlp.tokenization.utils import count_word_freqs, split_words

from .soln_tokenizers import (
    merges_0_expanded,
    text_0,
    tokens_0,
    vocab_0,
    vocab_0_expanded,
)
from .soln_utils import corpus

word_freqs = count_word_freqs(corpus, TOKENIZER)
word_bytes = split_words(word_freqs.keys(), byte_pair_str)


def test_count_pair_freqs():
    """
    tests count_pair_freqs()
    """
    pair_freqs = count_pair_freqs(word_freqs, word_bytes)
    assert max(pair_freqs, key=pair_freqs.get) == ("Ġ", "t")


def test_expand_vocab():
    """
    tests expand_vocab()
    """
    vocab_0_copy = vocab_0.copy()
    merges = {}
    expand_vocab(vocab_0_copy, word_freqs.copy(), word_bytes.copy(), merges, 50)
    assert vocab_0_copy == vocab_0_expanded
    assert merges == merges_0_expanded


def test_init_vocab():
    """
    tests init_vocab()
    """
    assert init_vocab(word_freqs.keys()) == vocab_0


def test_tokenize():
    """
    tests tokenize()
    """
    vocab_0_copy = vocab_0.copy()
    merges = {}
    expand_vocab(vocab_0_copy, word_freqs.copy(), word_bytes.copy(), merges, 50)
    assert tokenize(text_0, merges) == tokens_0
