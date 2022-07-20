from machine_learning.nlp.strings import word_piece_str
from machine_learning.nlp.tokenization.utils import count_word_freqs, split_words
from machine_learning.nlp.tokenization.word_piece import (
    TOKENIZER,
    expand_vocab,
    init_vocab,
    score_pairs,
    tokenize,
)

from .soln_tokenizers import text_1, tokens_1, vocab_1, vocab_1_expanded
from .soln_utils import corpus

word_freqs = count_word_freqs(corpus, TOKENIZER)
word_bytes = split_words(word_freqs.keys(), word_piece_str)


def test_expand_vocab():
    """
    tests expand_vocab()
    """
    vocab_1_copy = vocab_1.copy()
    expand_vocab(vocab_1_copy, word_freqs.copy(), word_bytes.copy(), 70)
    assert vocab_1_copy == vocab_1_expanded


def test_init_vocab():
    """
    tests init_vocab()
    """
    assert init_vocab(word_freqs.keys()) == vocab_1


def test_score_pairs():
    """
    tests score_pairs()
    """
    pair_scores = score_pairs(word_freqs, word_bytes)
    assert max(pair_scores, key=pair_scores.get) == ("a", "##b")


def test_tokenize():
    """
    tests tokenize()
    """
    vocab_1_copy = vocab_1.copy()
    expand_vocab(vocab_1_copy, word_freqs.copy(), word_bytes.copy(), 70)
    assert tokenize(text_1, vocab_1_copy) == tokens_1
