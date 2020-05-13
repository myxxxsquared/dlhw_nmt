import gzip
import os
import pickle
import random
import tensorflow_datasets as tfds
import numpy as np
import tensorflow as tf


def loaddata(filename):
    dataset = []
    for line in open(filename):
        line = line.strip()
        if not line:
            continue
        e, c, _ = line.split("\t")
        dataset.append((e, c))

    random.shuffle(dataset)

    alllen = len(dataset)
    trainlen = int(alllen * 0.7)
    validlen = int(alllen * 0.9)

    train = dataset[:trainlen]
    validation = dataset[trainlen:validlen]
    test = dataset[validlen:]

    train = list(zip(*train))
    validation = list(zip(*validation))
    test = list(zip(*test))

    return train, validation, test


def loadtokenizer(train):
    en, cn = train
    entok = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        en, target_vocab_size=10000
    )
    cntok = tfds.features.text.SubwordTextEncoder.build_from_corpus(
        cn, target_vocab_size=10000
    )
    return entok, cntok


def loadtokenized(
    dataset,
    entok,
    entok_begin,
    entok_end,
    cntok,
    cntok_begin,
    cntok_end,
):
    en, cn = dataset
    assert len(en) == len(cn)
    lendataset = len(en)

    SEQLEN = 20

    entoked = np.empty(dtype=np.int, shape=(lendataset, SEQLEN))
    cntoked = np.empty(dtype=np.int, shape=(lendataset, SEQLEN))

    for i, (sen, scn) in enumerate(zip(en, cn)):
        sen = [entok_begin] + entok.encode(sen) + [entok_end]
        if len(sen) < SEQLEN:
            sen = sen + [0] * (SEQLEN - len(sen))
        else:
            sen = sen[:SEQLEN]
        sen = np.array(sen)
        entoked[i] = sen

        scn = [cntok_begin] + cntok.encode(scn) + [cntok_end]
        if len(scn) < SEQLEN:
            scn = scn + [0] * (SEQLEN - len(scn))
        else:
            scn = scn[:SEQLEN]
        scn = np.array(scn)
        cntoked[i] = scn

    return entoked, cntoked


class DataObject:
    def __init__(
        self,
        train,
        validation,
        test,
        entok,
        entok_begin,
        entok_end,
        cntok,
        cntok_begin,
        cntok_end,
        train_tok,
        validation_tok,
        test_tok,
    ):
        self.train = train
        self.validation = validation
        self.test = test
        self.entok = entok
        self.entok_begin = entok_begin
        self.entok_end = entok_end
        self.cntok = cntok
        self.cntok_begin = cntok_begin
        self.cntok_end = cntok_end
        self.train_tok = train_tok
        self.validation_tok = validation_tok
        self.test_tok = test_tok

        self.data_train = tf.data.Dataset.from_tensor_slices(self.train_tok).shuffle(buffer_size=1024).batch(512)
        self.data_validation = tf.data.Dataset.from_tensor_slices(self.validation_tok).batch(512)
        self.data_test = tf.data.Dataset.from_tensor_slices(self.test_tok).batch(512)

def preprocessdata():
    PATH_ORIGIN = "data/cmn.txt"
    PATH_RAW = "data/raw.pkl.gz"
    PATH_TOKEN_EN = "data/entok"
    PATH_TOKEN_EN_FILE = PATH_TOKEN_EN + ".subwords"
    PATH_TOKEN_CN = "data/cntok"
    PATH_TOKEN_CN_FILE = PATH_TOKEN_CN + ".subwords"
    PATH_TOKENED = "data/toked.pkl.gz"

    if os.path.exists(PATH_RAW):
        train, validation, test = pickle.load(gzip.open(PATH_RAW, "rb"))
    else:
        train, validation, test = loaddata(PATH_ORIGIN)
        pickle.dump((train, validation, test), gzip.open(PATH_RAW, "wb"))

    if os.path.exists(PATH_TOKEN_EN_FILE) and os.path.exists(PATH_TOKEN_CN_FILE):
        entok = tfds.features.text.SubwordTextEncoder.load_from_file(PATH_TOKEN_EN)
        cntok = tfds.features.text.SubwordTextEncoder.load_from_file(PATH_TOKEN_CN)
    else:
        entok, cntok = loadtokenizer(train)
        entok.save_to_file(PATH_TOKEN_EN)
        cntok.save_to_file(PATH_TOKEN_CN)

    entok_begin = entok.vocab_size
    entok_end = entok.vocab_size + 1
    cntok_begin = cntok.vocab_size
    cntok_end = cntok.vocab_size + 1

    if os.path.exists(PATH_TOKENED):
        train_tok, validation_tok, test_tok = pickle.load(gzip.open(PATH_TOKENED, "rb"))
    else:
        train_tok = loadtokenized(
            train,
            entok,
            entok_begin,
            entok_end,
            cntok,
            cntok_begin,
            cntok_end,
        )
        validation_tok = loadtokenized(
            validation,
            entok,
            entok_begin,
            entok_end,
            cntok,
            cntok_begin,
            cntok_end,
        )
        test_tok = loadtokenized(
            test,
            entok,
            entok_begin,
            entok_end,
            cntok,
            cntok_begin,
            cntok_end,
        )
        pickle.dump(
            (train_tok, validation_tok, test_tok), gzip.open(PATH_TOKENED, "wb")
        )

    return DataObject(
        train,
        validation,
        test,
        entok,
        entok_begin,
        entok_end,
        cntok,
        cntok_begin,
        cntok_end,
        train_tok,
        validation_tok,
        test_tok,
    )


__all__ = ["DataObject", "preprocessdata"]

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    data = preprocessdata()

    print('CnToken:')
    tok = data.cntok
    for i in range(10):
        print(f"{i}: {repr(tok.decode([i]))}")

    print()
    print('EnToken:')
    tok = data.entok
    for i in range(10):
        print(f"{i}: {repr(tok.decode([i]))}")