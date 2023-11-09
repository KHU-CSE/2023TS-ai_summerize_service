import nltk
from .normalizer import *

nltk.download('punkt')
from nltk.tokenize import sent_tokenize

def split_context_into_sentences(text):
    sentences = sent_tokenize(text)
    sentences = [normalize(sentence) for sentence in sentences]
    return sentences

