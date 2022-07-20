from machine_learning.nlp.strings import end_of_text_str
from machine_learning.nlp.tokenization.word_piece import VOCAB

vocab_0 = {
    end_of_text_str,
    ",",
    ".",
    "C",
    "F",
    "H",
    "T",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "y",
    "z",
    "Ġ",
}
vocab_0_expanded = {
    end_of_text_str,
    ",",
    ".",
    "C",
    "F",
    "H",
    "T",
    "a",
    "b",
    "c",
    "d",
    "e",
    "f",
    "g",
    "h",
    "i",
    "k",
    "l",
    "m",
    "n",
    "o",
    "p",
    "r",
    "s",
    "t",
    "u",
    "v",
    "w",
    "y",
    "z",
    "Ġ",
    "Ġt",
    "is",
    "er",
    "Ġa",
    "Ġto",
    "en",
    "Th",
    "This",
    "ou",
    "se",
    "Ġtok",
    "Ġtoken",
    "nd",
    "Ġis",
    "Ġth",
    "Ġthe",
    "in",
    "Ġab",
    "Ġtokeni",
}
merges_0_expanded = {
    ("Ġ", "t"): "Ġt",
    ("i", "s"): "is",
    ("e", "r"): "er",
    ("Ġ", "a"): "Ġa",
    ("Ġt", "o"): "Ġto",
    ("e", "n"): "en",
    ("T", "h"): "Th",
    ("Th", "is"): "This",
    ("o", "u"): "ou",
    ("s", "e"): "se",
    ("Ġto", "k"): "Ġtok",
    ("Ġtok", "en"): "Ġtoken",
    ("n", "d"): "nd",
    ("Ġ", "is"): "Ġis",
    ("Ġt", "h"): "Ġth",
    ("Ġth", "e"): "Ġthe",
    ("i", "n"): "in",
    ("Ġa", "b"): "Ġab",
    ("Ġtoken", "i"): "Ġtokeni",
}

vocab_1_init = {
    "##a",
    "##b",
    "##c",
    "##d",
    "##e",
    "##f",
    "##g",
    "##h",
    "##i",
    "##k",
    "##l",
    "##m",
    "##n",
    "##o",
    "##p",
    "##r",
    "##s",
    "##t",
    "##u",
    "##v",
    "##w",
    "##y",
    "##z",
    ",",
    ".",
    "C",
    "F",
    "H",
    "T",
    "a",
    "b",
    "c",
    "g",
    "h",
    "i",
    "s",
    "t",
    "u",
    "w",
    "y",
}
vocab_1_init_expanded = {
    "##a",
    "##b",
    "##c",
    "##d",
    "##e",
    "##f",
    "##g",
    "##h",
    "##i",
    "##k",
    "##l",
    "##m",
    "##n",
    "##o",
    "##p",
    "##r",
    "##s",
    "##t",
    "##u",
    "##v",
    "##w",
    "##y",
    "##z",
    ",",
    ".",
    "C",
    "F",
    "H",
    "T",
    "a",
    "b",
    "c",
    "g",
    "h",
    "i",
    "s",
    "t",
    "u",
    "w",
    "y",
    "ab",
    "##fu",
    "Fa",
    "Fac",
    "##ct",
    "##ful",
    "##full",
    "##fully",
    "Th",
    "ch",
    "##hm",
    "cha",
    "chap",
    "chapt",
    "##thm",
    "Hu",
    "Hug",
    "Hugg",
    "sh",
    "th",
    "is",
    "##thms",
    "##za",
    "##zat",
    "##ut",
}
vocab_1 = VOCAB.union(vocab_1_init)
vocab_1_expanded = VOCAB.union(vocab_1_init_expanded)

text_0 = "This is not a token."
tokens_0 = ["This", "Ġis", "Ġ", "n", "o", "t", "Ġa", "Ġtoken", "."]

text_1 = "This is the Hugging Face course!"
tokens_1 = [
    "Th",
    "##i",
    "##s",
    "is",
    "th",
    "##e",
    "Hugg",
    "##i",
    "##n",
    "##g",
    "Fac",
    "##e",
    "c",
    "##o",
    "##u",
    "##r",
    "##s",
    "##e",
    "[UNK]",
]