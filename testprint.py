import pickle
import gzip

new_corpus_cn, new_corpus_en = pickle.load(gzip.open('data/new_corpus.pkl.gz', 'rb'))

print(new_corpus_cn[0])
print(new_corpus_en[0])
