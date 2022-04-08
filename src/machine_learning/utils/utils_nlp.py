import re
from typing import List, Optional

from spacy.lang.en import English


def lemmatize(
    tokenized_doc: List[str], nlp: English, allowed_postags: Optional[List[str]] = None
) -> List[str]:
    """
    lemmatize tokens in doc
    :param tokenized_doc: doc tokens
    :param nlp: spacy language English model
    :param allowed_postags: postags allowed
    :return:
    """
    if allowed_postags is None:
        allowed_postags = ["NOUN", "ADJ", "VERB", "ADV"]
    doc = nlp(" ".join(tokenized_doc))
    return [token.lemma_ for token in doc if token.pos_ in allowed_postags]


def remove_email(sent: str) -> str:
    """
    removes abc@xyz.com from sentence
    :param sent: sentence
    :return: sentence without abc@xyz.com
    """
    return re.sub(r"\S*@\S*\s?", "", sent)


def remove_newline_char(sent: str, newline_char: str = r"\s+") -> str:
    """
    removes newline character from sentence
    :param sent: sentence
    :param newline_char: newline character
    :return: sentence without newline character
    """
    return re.sub(newline_char, " ", sent)


def remove_single_quote(sent: str) -> str:
    """
    removes single quotes from sentence
    :param sent: sentence
    :return: sentence without single quotes
    """
    return re.sub("'", "", sent)


def remove_stopwords(doc: List[str], stopwords: List[str]) -> List[str]:
    """
    removes stopwords from doc
    :param doc: doc
    :param stopwords: stopwords
    :return: doc without stopwords
    """
    return [word for word in doc if word not in stopwords]
