import spacy

from machine_learning.utils.utils_nlp import (
    lemmatize,
    remove_email,
    remove_newline_char,
    remove_single_quote,
    remove_stopwords,
)

nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

word_0 = "sender"
word_1 = "send"
word_2 = "sending"
word_3 = "sent"
word_4 = "stopword"
word_5 = "to"
word_6 = "'"
word_7 = "ABC"
word_8 = "'"
word_9 = "at"
word_10 = "abc@gmail.com"
word_11 = "please"
word_12 = "\n"
doc = f"{word_0} {word_1} {word_2} {word_3} {word_4} {word_5} {word_6}{word_7}{word_8} {word_9} {word_10} {word_11} {word_12}"
doc_without_email = f"{word_0} {word_1} {word_2} {word_3} {word_4} {word_5} {word_6}{word_7}{word_8} {word_9} {word_11} {word_12}"
doc_without_newline_char = f"{word_0} {word_1} {word_2} {word_3} {word_4} {word_5} {word_6}{word_7}{word_8} {word_9} {word_10} {word_11} "
doc_without_single_quote = f"{word_0} {word_1} {word_2} {word_3} {word_4} {word_5} {word_7} {word_9} {word_10} {word_11} {word_12}"
doc_tokenized = ["sender", "send", "sent", "sending"]
doc_tokenized_without_sender = ["send", "sent", "sending"]


def test_lemmatize():
    """
    tests lemmatize()
    """
    lemmatized_doc = lemmatize(doc_tokenized, nlp)
    assert lemmatized_doc == ["sender", "send", "send", "send"]


def test_remove_email():
    """
    tests remove_email()
    """
    assert remove_email(doc) == doc_without_email


def test_remove_newline_char():
    """
    tests remove_newline_char()
    """
    assert remove_newline_char(doc) == doc_without_newline_char


def test_remove_single_quote():
    """
    tests remove_single_quote()
    """
    assert remove_single_quote(doc) == doc_without_single_quote


def test_remove_stopwords():
    """
    tests remove_stopwords()
    """
    assert remove_stopwords(doc_tokenized, [word_0]) == doc_tokenized_without_sender
    print(type(nlp))
