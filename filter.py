from PIL import Image, ImageFont, ImageDraw
import torch
import numpy
from pprint import pprint

fnt = ImageFont.truetype("D2Coding-Ver1.3.2-20180524.ttf", 16, encoding="UTF-8")


def normalize(t: str):
    im = Image.new("1", (384, 16), (1,))  # White
    dr = ImageDraw.Draw(im)
    dr.text((0, 0), t, font=fnt, fill=(0,))
    return numpy.asarray(numpy.float32(im.split()[0]))


pprint(normalize(u"가나다라ㄱㄴㄷㄹ!@#!@$#@!$@1123576854&%^|"))