import numpy as np
import operator
from collections import defaultdict
from nltk.tokenize import word_tokenize


class GloveVectorizer:

    def __init__(self, fname):

        self.embedding_size = None
        self.max_seq_length = None
        self.embeddings = dict()

        f = open(fname)
        for line in f:
            values = line.split()
            word = values[0]

            coefs = np.asarray(values[1:], dtype='float32')
            if self.embedding_size is None:
                self.embedding_size = len(coefs)

            self.embeddings[word] = coefs
        f.close()

        self.tokenize = word_tokenize

        # add zeros vector for padding and random for oov (out of vocabulary words)
        self.embeddings["__PADDING__"] = np.zeros(self.embedding_size, dtype='float32')
        self.embeddings["__OOV_WORD__"] = np.random.uniform(-1, 1, self.embedding_size)
        self.pad_id = 0
        self.oov_id = 1
        self.base_word_id = 2

        self.word_index = None
        self.word_counts = None
        self.doc_counts = None
        self.inverse_word_index = None
        self.trimmed_word_index = None
        self.weights = None

    def fit(self, X, D, seq_length=None, wgt=1, wgt_inverse=0):

        max_seq_length = 0
        word_counts = defaultdict(lambda: 0)
        doc_counts = defaultdict(lambda: 1)
        for doc in X:
            sl = 0
            tokens = self.tokenize(doc)

            for w in set(tokens):
                doc_counts[w] += 1

            for w in tokens:
                sl += 1
                word_counts[w] += 1
            max_seq_length = sl if sl > max_seq_length else max_seq_length

        self.max_seq_length = max_seq_length

        word_counts = dict(word_counts)
        word_counts = sorted(word_counts.items(), reverse=True, key=operator.itemgetter(1))

        doc_words = doc_counts.keys()

        doc_ids = np.array(list(doc_counts.values())) / float(X.shape[0])
        self.doc_counts = dict(zip(doc_words, doc_ids))
        self.doc_counts["__PADDING__"] = 0
        self.doc_counts["__OOV_WORD__"] = 0

        self.word_index = dict()
        self.word_index["__PADDING__"] = self.pad_id
        self.word_index["__OOV_WORD__"] = self.oov_id

        self.word_counts = dict()
        for i, (w, c) in enumerate(word_counts):
            self.word_index[w] = i + 1 + self.base_word_id
            self.word_counts[i + 1 + self.base_word_id] = c

        self.inverse_word_index = {v: k for k, v in self.word_index.items()}

        from nltk.stem.snowball import EnglishStemmer
        stemmer = EnglishStemmer()
        V = []
        for i, x in enumerate(X):
            deplist = df["dependency_path"].values[i]
            tokens = word_tokenize(x)

            if seq_length is not None:
                v = np.ones(seq_length) * wgt_inverse
            else:
                v = np.ones(self.max_seq_length) * wgt_inverse

            for j, w in enumerate(tokens):
                if stemmer.stem(w.lower()) in deplist or w.lower() in deplist:
                    v[j] = wgt
            V.append(v)

        self.weights = np.array(V)

        return V, self.weights

    def transform_id(self, X, max_vocabulary=None, seq_length=None):

        if seq_length is None:
            vX = np.zeros((X.shape[0], self.max_seq_length), dtype=int) * self.pad_id
        else:
            vX = np.zeros((X.shape[0], seq_length), dtype=int) * self.pad_id

        for i, doc in enumerate(X):
            for j, w in enumerate(self.tokenize(doc)):
                # respect max seq length
                if j >= vX.shape[1]:
                    break

                if w in self.word_index:
                    wid = self.word_index[w]

                    # check if oov
                    if max_vocabulary is not None and wid > max_vocabulary:
                        wid = self.oov_id

                    else:
                        # out of index vocabulary
                        wid = self.word_index[w]
                else:
                    wid = self.oov_id

                vX[i, j] = wid

        return vX

    def transform_sumembed(self, X, max_vocabulary=None, seq_length=None, idf=False, weights=False):

        if seq_length is None or seq_length > self.max_seq_length:
            seq_length = self.max_seq_length

        X_ids = self.transform_id(X, max_vocabulary, seq_length)
        X_emb = np.zeros((X.shape[0], self.embedding_size))

        for i, x in enumerate(X_ids):
            sent_emb = np.zeros((seq_length, self.embedding_size))
            for j, wid in enumerate(x):
                w = self.inverse_word_index[wid]
                if w in self.embeddings:

                    sent_emb[j] = self.embeddings[self.inverse_word_index[wid]]

                    if idf:
                        sent_emb[j] = sent_emb[j] * self.doc_counts[w]

                    if weights:
                        sent_emb[j] = sent_emb[j] * self.weights[i][j]

                else:
                    sent_emb[j] = self.embeddings[self.inverse_word_index[self.oov_id]]

                    if idf:
                        sent_emb[j] = sent_emb[j] * self.doc_counts[w]

                    if weights:
                        sent_emb[j] = sent_emb[j] * self.weights[i][j]

            sent_emb = sent_emb.sum(axis=0)

            if weights:
                dep_path_length = len([k for k in self.weights[i] if k != 0])
                if dep_path_length == 0:
                    print(i)
                sent_emb /= float(dep_path_length + 1)
            else:
                actual_length = len([l for l in x if l != self.pad_id])
                sent_emb /= float(actual_length)

            X_emb[i] = sent_emb

        return X_emb
