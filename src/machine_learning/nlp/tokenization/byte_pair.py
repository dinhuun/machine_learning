"""
from huggingface course
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple

from transformers import AutoTokenizer

from machine_learning.nlp.strings import byte_pair_str, end_of_text_str, gpt2_str
from machine_learning.nlp.tokenization.utils import merge_pair

TOKENIZER = AutoTokenizer.from_pretrained(gpt2_str)


def count_pair_freqs(
    word_freqs: Dict[str, int], word_bytes: Dict[str, List[str]]
) -> Dict[Tuple[str, str], int]:
    """
    counts frequency for each pair of bytes (byte_0, byte_1) that appear in each word based on that word's frequency
    :param word_freqs: words and their frequencies
    :param word_bytes: words and their bytes
    :return: pairs and their frequencies
    """
    pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)
    for word, freq in word_freqs.items():
        bytes_ = word_bytes[word]
        if len(bytes_) == 1:
            continue
        for i in range(len(bytes_) - 1):
            pair = (bytes_[i], bytes_[i + 1])
            pair_freqs[pair] += freq
    return pair_freqs


def expand_vocab(
    vocab: Set[str],
    word_freqs: Dict[str, int],
    word_bytes: Dict[str, List[str]],
    merges: Optional[Dict[Tuple[str, str], str]] = None,
    vocab_size: int = 100,
):
    """
    expands vocabulary with mergers of most frequent pairs until desired vocabulary size is reached
    :param vocab: vocabulary
    :param word_freqs: words and their frequencies
    :param word_bytes: words and their bytes
    :param merges: pairs and their mergers
    :param vocab_size: desired vocabulary size
    :return: vocab and merges may have been mutated
    """
    if merges is None:
        merges = {}

    prev_pair = ("", "")
    while len(vocab) < vocab_size:
        pair_freqs = count_pair_freqs(word_freqs, word_bytes)
        pair = max(pair_freqs, key=pair_freqs.get)  # type: ignore

        if pair == prev_pair:
            print(f"can not expand current vocabulary of size {len(vocab)} further")
            break

        merger = merge_pair(*pair, word_bytes, byte_pair_str)
        merges[pair] = merger
        vocab.add(merger)
        prev_pair = pair


def init_vocab(words: Iterable[str]) -> Set[str]:
    """
    initializes vocabulary from words plus "<|endoftext|>"
    """
    vocab = {end_of_text_str}
    vocab.update(*map(set, words))  # type: ignore
    return vocab


def tokenize(
    text: str, merges: Dict[Tuple[str, str], str], tokenizer=TOKENIZER
) -> List[str]:
    """
    tokenizes text "word_0 word_1..." into [token_0, token_1,...] according to merges
    """
    pre_tokenized_rslt = tokenizer._tokenizer.pre_tokenizer.pre_tokenize_str(text)
    pre_tokenized_text = [word for word, _ in pre_tokenized_rslt]
    word_bytes = [list(word) for word in pre_tokenized_text]
    for pair, merger in merges.items():
        for idx, bytes_ in enumerate(word_bytes):
            i = 0
            while i < len(bytes_) - 1:
                if bytes_[i] == pair[0] and bytes_[i + 1] == pair[1]:
                    bytes_ = bytes_[:i] + [merger] + bytes_[i + 2 :]
                else:
                    i += 1
            word_bytes[idx] = bytes_
    return sum(word_bytes, [])
