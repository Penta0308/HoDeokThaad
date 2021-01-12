import json
import random
import re

import torch
import numpy as np
from konlpy.tag import Komoran


def config_get(key):
    with open("config.json", "r", encoding="UTF-8") as f:
        settings = json.load(f)
    try:
        return settings[key]
    except KeyError:
        config_update(key, None)
        return None

def config_update(key, val):
    with open("config.json", "r", encoding="UTF-8") as f:
        settings = json.load(f)
    settings[key] = val
    with open("config.json", "w", encoding="UTF-8") as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)


komoran = Komoran(userdic="userdic.tsv")

SEED = 1

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

FILTER_SIZES = [2, 3, 4, 5, 6, 7]


def tokenizer(text):
    token = [t for t in komoran.morphs(text)]
    if len(token) < max(FILTER_SIZES):
        for i in range(0, max(FILTER_SIZES) - len(token)):
            token.append('<PAD>')
    return token


def url_filter(text):
    re_pattern = r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),|]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    new_text = re.sub(re_pattern, 'url', text)
    return new_text


def price_filter(text):
    re_pattern = r'\d{1,3}[,\.]\d{1,3}[만\천]?\s?[원]|\d{1,5}[만\천]?\s?[원]'
    text = re.sub(re_pattern, 'money', text)
    re_pattern = r'[일/이/삼/사/오/육/칠/팔/구/십/백][만\천]\s?[원]'
    text = re.sub(re_pattern, 'money', text)
    re_pattern = r'(?!-)\d{2,4}[0]{2,4}(?!년)(?!.)|\d{1,3}[,/.]\d{3}'
    new_text = re.sub(re_pattern, 'money', text)
    return new_text