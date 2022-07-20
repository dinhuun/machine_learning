import pytest
from transformers import AutoTokenizer

from machine_learning.nlp.strings import (
    bert_base_cased_str,
    byte_pair_str,
    gpt2_str,
    word_piece_str,
)
from machine_learning.nlp.tokenization.utils import (
    count_byte_freqs_pair_freqs,
    count_word_freqs,
    merge_pair,
    split_words,
)
from machine_learning.strings import not_implemented_str

from .soln_utils import (
    byte_freqs,
    corpus,
    pair_freqs,
    word_bytes,
    word_bytes_0,
    word_bytes_1,
    word_bytes_after_merge,
    word_freqs,
    word_freqs_0,
    word_freqs_1,
    words,
)


def test_count_byte_freqs_pair_freqs():
    """
    tests count_byte_freqs_pair_freqs()
    """
    assert count_byte_freqs_pair_freqs(word_freqs, word_bytes) == (
        byte_freqs,
        pair_freqs,
    )


def test_count_word_freqs():
    """
    tests count_word_freqs()
    """
    tokenizer_0 = AutoTokenizer.from_pretrained(gpt2_str)
    assert count_word_freqs(corpus, tokenizer_0) == word_freqs_0

    tokenizer_1 = AutoTokenizer.from_pretrained(bert_base_cased_str)
    assert count_word_freqs(corpus, tokenizer_1) == word_freqs_1


def test_merge_pair():
    """
    tests merge_pair()
    """
    with pytest.raises(NotImplementedError):
        merge_pair("", "", {}, not_implemented_str)

    word_bytes_copy_0 = word_bytes.copy()
    assert merge_pair("b", "c", word_bytes_copy_0, byte_pair_str) == "bc"
    assert word_bytes_copy_0 == word_bytes_after_merge

    word_bytes_copy_1 = word_bytes.copy()
    assert merge_pair("b", "c", word_bytes_copy_1, word_piece_str) == "bc"
    assert word_bytes_copy_1 == word_bytes_after_merge


def test_split_words():
    """
    tests split_words()
    """
    with pytest.raises(NotImplementedError):
        split_words(words, not_implemented_str)

    assert split_words(words, byte_pair_str) == word_bytes_0

    assert split_words(words, word_piece_str) == word_bytes_1
