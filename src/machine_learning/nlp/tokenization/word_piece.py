"""
from huggingface course
"""

from typing import Dict, Iterable, List, Set, Tuple

from transformers import AutoTokenizer

from machine_learning.nlp.strings import bert_base_cased_str, unk_str, word_piece_str
from machine_learning.nlp.tokenization.utils import (
    count_byte_freqs_pair_freqs,
    merge_pair,
)

TOKENIZER = AutoTokenizer.from_pretrained(bert_base_cased_str)
VOCAB = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"}


def encode_word(word: str, vocab: Set[str]) -> List[str]:
    """
    encodes word into [token_0, token_1,...] using tokens in vocabulary
    """
    tokens = []
    while len(word) > 0:
        i = len(word)
        while i > 0 and word[:i] not in vocab:
            i -= 1
        if i == 0:
            return [unk_str]
        else:
            tokens.append(word[:i])
            word = word[i:]
            if len(word) > 0:
                word = "##" + word
    return tokens


def expand_vocab(
    vocab: Set[str],
    word_freqs: Dict[str, int],
    word_bytes: Dict[str, List[str]],
    vocab_size: int = 100,
):
    """
    expands vocabulary with mergers of highest scored pairs until desired vocabulary size is reached
    :param vocab: vocabulary
    :param word_freqs: words and their frequencies
    :param word_bytes: words and their bytes
    :param vocab_size: desired vocabulary size
    :return: vocab may have been mutated
    """
    prev_pair = ("", "")
    while len(vocab) < vocab_size:
        pair_scores = score_pairs(word_freqs, word_bytes)
        pair = max(pair_scores, key=pair_scores.get)  # type: ignore

        if pair == prev_pair:
            print(f"can not expand current vocabulary of size {len(vocab)} further")
            break

        merged = merge_pair(pair[0], pair[1], word_bytes, word_piece_str)
        vocab.add(merged)
        prev_pair = pair


def init_vocab(words: Iterable[str]) -> Set[str]:
    """
    initializes vocabulary from words plus "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]
    """
    vocab = VOCAB.copy()
    for word in words:
        for i, byte in enumerate(word):
            if i == 0:
                vocab.add(byte)
            else:
                vocab.add("##" + byte)
    return vocab


def score_pairs(
    word_freqs: Dict[str, int], word_bytes: Dict[str, List[str]]
) -> Dict[Tuple[str, str], float]:
    """
    scores each pair of bytes (byte_0, byte_1) as (its frequency) / (frequency of byte_0 * frequency of byte_1)
    :param word_freqs: words and their frequencies
    :param word_bytes: words and their bytes
    :return: pairs and their scores
    """
    byte_freqs, pair_freqs = count_byte_freqs_pair_freqs(word_freqs, word_bytes)
    pair_scores = {}
    for pair, freq in pair_freqs.items():
        pair_scores[pair] = freq / (byte_freqs[pair[0]] * byte_freqs[pair[1]])
    return pair_scores


def tokenize(text: str, vocab: Set[str], tokenizer=TOKENIZER) -> List[str]:
    """
    tokenizes text "word_0 word_1..." into [token_0, token_1,...] using tokens in vocabulary
    """
    pre_tokenized_rslt = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, _ in pre_tokenized_rslt]
    encoded_words = [encode_word(word, vocab) for word in pre_tokenized_text]
    return sum(encoded_words, [])
