import os
import json

import responder
import numpy as np
from bpemb import BPEmb


env = os.environ
DEBUG = env['DEBUG'] in ['1', 'True', 'true']
LANG = env['LANG']
DIM = int(env['DIM'])
VS = int(env['VS'])

api = responder.API(debug=DEBUG)
bpemb = BPEmb(lang=LANG, dim=DIM, vs=VS)


def get_subwords(text):
    return bpemb.encode(text)


def get_emb(text):
    vectors = bpemb.embed(text)
    return np.mean(vectors, axis=0).tolist()


def get_subwords_and_emb(text):
    subwords = get_subwords(text)
    emb = get_emb(text)
    return dict(subwords=subwords, embedding=emb)


@api.route("/")
async def encode(req, resp):
    body = await req.text
    texts = json.loads(body)
    data_dicts = [get_subwords_and_emb(text) for text in texts]
    resp_dict = dict(data=data_dicts, dim=DIM)
    resp.media = resp_dict


if __name__ == "__main__":
    api.run()