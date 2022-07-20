from collections import defaultdict
from typing import Dict, Iterable, List, Tuple


def count_byte_freqs_pair_freqs(
    word_freqs: Dict[str, int], word_bytes: Dict[str, List[str]]
) -> Tuple[Dict, Dict]:
    """
    counts
        * frequency for each byte
        * frequency for each pair of bytes (byte_0, byte_1)
    that appear in each word based on that word's frequency
    :param word_freqs: words and their frequencies
    :param word_bytes: words and their bytes
    :return: {byte: its frequency}, {pair of bytes: its frequency}
    """
    byte_freqs: Dict[str, int] = defaultdict(int)
    pair_freqs: Dict[Tuple[str, str], int] = defaultdict(int)

    for word, freq in word_freqs.items():
        bytes_ = word_bytes[word]
        n = len(bytes_)
        if n == 1:
            byte_freqs[bytes_[0]] += freq
        else:
            for i in range(n - 1):
                byte_freqs[bytes_[i]] += freq
                pair = (bytes_[i], bytes_[i + 1])
                pair_freqs[pair] += freq
            byte_freqs[bytes_[-1]] += freq
    return byte_freqs, pair_freqs


def count_word_freqs(corpus: List[str], tokenizer) -> Dict[str, int]:
    """
    counts word frequencies in corpus
    :param corpus: list of texts
    :param tokenizer: tokenizer to pretokenize each text in corpus
    :return: words and their frequencies
    """
    word_freqs: Dict[str, int] = defaultdict(int)
    for text in corpus:
        words_with_offsets = tokenizer.backend_tokenizer.pre_tokenizer.pre_tokenize_str(
            text
        )
        for word, _ in words_with_offsets:
            word_freqs[word] += 1
    return word_freqs


def merge_pair(
    byte_0: str, byte_1: str, word_bytes: Dict[str, List[str]], tokenizer_str: str
) -> str:
    """
    merges pair of bytes (byte_0, byte_1) that appear in each word's bytes repeatedly
    for Byte-Pair tokenizer or WordPiece tokenizer
    :param byte_0: byte_0
    :param byte_1: byte_1
    :param word_bytes: words and their bytes
    :param tokenizer_str: either "byte_pair" or "word_piece"
    :return: merger of byte_0 and byte_1, while word_bytes may have been mutated
    """
    if tokenizer_str == "byte_pair":
        merger = byte_0 + byte_1
    elif tokenizer_str == "word_piece":
        merger = byte_0 + byte_1[2:] if byte_1.startswith("##") else byte_0 + byte_1
    else:
        raise NotImplementedError(
            f"merging for {tokenizer_str} has not been implemented"
        )

    for word in word_bytes:
        bytes_ = word_bytes[word]
        if len(bytes_) == 1:
            continue

        i = 0
        while i < len(bytes_) - 1:
            if bytes_[i] == byte_0 and bytes_[i + 1] == byte_1:
                bytes_ = bytes_[:i] + [merger] + bytes_[i + 2 :]
            else:
                i += 1
        word_bytes[word] = bytes_
    return merger


def split_words(words: Iterable[str], tokenizer_str: str) -> Dict[str, List[str]]:
    """
    splits each word into its bytes for Byte-Pair tokenizer or WordPiece tokenizer
    """
    word_bytes = defaultdict(list)
    if tokenizer_str == "byte_pair":
        for word in words:
            word_bytes[word] = list(word)
    elif tokenizer_str == "word_piece":
        for word in words:
            for i, byte in enumerate(word):
                if i == 0:
                    word_bytes[word].append(byte)
                else:
                    word_bytes[word].append(f"##{byte}")
    else:
        raise NotImplementedError(
            f"splitting words for {tokenizer_str} has not been implemented"
        )
    return word_bytes
