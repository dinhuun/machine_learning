corpus = [
    "This is the Hugging Face Course.",
    "This chapter is about tokenization.",
    "This section shows several tokenizer algorithms.",
    "Hopefully, you will be able to understand how they are trained and generate tokens.",
]

words = ["abc", "bcd"]
word_freqs = {
    "abc": 1,
    "bcd": 1,
}
word_bytes = {
    "abc": ["a", "b", "c"],
    "bcd": ["b", "c", "d"],
}
word_bytes_after_merge = {"abc": ["a", "bc"], "bcd": ["bc", "d"]}
byte_freqs = {"a": 1, "b": 2, "c": 2, "d": 1}
pair_freqs = {("a", "b"): 1, ("b", "c"): 2, ("c", "d"): 1}

word_bytes_0 = {
    "abc": ["a", "b", "c"],
    "bcd": ["b", "c", "d"],
}
word_bytes_1 = {
    "abc": ["a", "##b", "##c"],
    "bcd": ["b", "##c", "##d"],
}

word_freqs_0 = {
    "This": 3,
    "Ġis": 2,
    "Ġthe": 1,
    "ĠHugging": 1,
    "ĠFace": 1,
    "ĠCourse": 1,
    ".": 4,
    "Ġchapter": 1,
    "Ġabout": 1,
    "Ġtokenization": 1,
    "Ġsection": 1,
    "Ġshows": 1,
    "Ġseveral": 1,
    "Ġtokenizer": 1,
    "Ġalgorithms": 1,
    "Hopefully": 1,
    ",": 1,
    "Ġyou": 1,
    "Ġwill": 1,
    "Ġbe": 1,
    "Ġable": 1,
    "Ġto": 1,
    "Ġunderstand": 1,
    "Ġhow": 1,
    "Ġthey": 1,
    "Ġare": 1,
    "Ġtrained": 1,
    "Ġand": 1,
    "Ġgenerate": 1,
    "Ġtokens": 1,
}

word_freqs_1 = {
    "This": 3,
    "is": 2,
    "the": 1,
    "Hugging": 1,
    "Face": 1,
    "Course": 1,
    ".": 4,
    "chapter": 1,
    "about": 1,
    "tokenization": 1,
    "section": 1,
    "shows": 1,
    "several": 1,
    "tokenizer": 1,
    "algorithms": 1,
    "Hopefully": 1,
    ",": 1,
    "you": 1,
    "will": 1,
    "be": 1,
    "able": 1,
    "to": 1,
    "understand": 1,
    "how": 1,
    "they": 1,
    "are": 1,
    "trained": 1,
    "and": 1,
    "generate": 1,
    "tokens": 1,
}
