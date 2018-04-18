# from gensim.test.utils import datapath, get_tmpfile
# from gensim.models import KeyedVectors
# glove_file = datapath('/home/b/Downloads/glove.840B.300d.txt')
# from gensim.scripts.glove2word2vec import glove2word2vec
# tmp_file = get_tmpfile("test_word2vec.txt")
# glove2word2vec(glove_file, tmp_file)
#
# model = KeyedVectors.load_word2vec_format(tmp_file)

import spacy
import requests
import pandas as pd
from portmanteau import bridge

nlp = spacy.load('en_core_web_sm')

# w = nlp(u'a hundred and fifty words make a book')[4]

# len(by_similarity)
# most_similar(w)
#
# import nltk
# from nltk.corpus import wordnet
# synonyms = []
# antonyms = []
#
# for syn in wordnet.synsets(wordnet.synset('ship.n.01')):
#     print syn
#     for l in syn.lemmas():
#         print "----", l
#         synonyms.append(l.name())
#
# w = wordnet.synset('word.n.01')
#
# for i, j in w.hypernym_distances():
#     print i.hyponyms()


def getsyns(word, param='spc'):
    r = requests.get('https://api.datamuse.com/words?rel_' +
                     param + '=' + word)
    return r.json()


"""
    jja 	Popular nouns modified by the given adjective, per Google Books Ngrams 	gradual → increase
    jjb 	Popular adjectives used to modify the given noun, per Google Books Ngrams 	beach → sandy
    syn 	Synonyms (words contained within the same WordNet synset) 	ocean → sea
    trg 	"Triggers" (words that are statistically associated with the query word in the same piece of text.) 	cow → milking
    ant 	Antonyms (per WordNet) 	late → early
    spc 	"Kind of" (direct hypernyms, per WordNet) 	gondola → boat
    gen 	"More general than" (direct hyponyms, per WordNet) 	boat → gondola
    com 	"Comprises" (direct holonyms, per WordNet) 	car → accelerator
    par 	"Part of" (direct meronyms, per WordNet) 	trunk → tree
    bga 	Frequent followers (w′ such that P(w′|w) ≥ 0.001, per Google Books Ngrams) 	wreak → havoc
    bgb 	Frequent predecessors (w′ such that P(w|w′) ≥ 0.001, per Google Books Ngrams) 	havoc → wreak
    rhy 	Rhymes ("perfect" rhymes, per RhymeZone) 	spade → aid
    nry 	Approximate rhymes (per RhymeZone) 	forest → chorus
    hom 	Homophones (sound-alike words) 	course → coarse
    cns 	Consonant match 	sample → simple
"""


def get_df(w):
    # param_list = ['spc', 'trg', 'gen', 'com', 'par']
    param_list = ['spc', 'trg']
    words = {}
    for p in param_list:
        for i in getsyns(w.lemma_, p):
            x = nlp(unicode(i['word']))
            for j in x:
                if j.lemma_ not in words:
                    words[j.lemma_] = {'sim': w.similarity(j)}
    if len(words) == 0:
        return None
    else:
        return pd.DataFrame(words).T.sort_values('sim', ascending=False)


def get_words(w, lim=10):
    result = get_df(w)
    return None if result is None else result[:lim].index


def get_wdump(desc="A friend for paying on websites."):
    word_dump = {}
    patterns = []
    for i in nlp(unicode(desc)):
        print i
        if not i.is_stop and i.pos_ in ['ADJ', 'NOUN', 'VERB']:
            syns = list(get_words(i, 5))
            syns = syns + [i.lemma_] if i.lemma_ not in syns else syns
            word_dump[i.lemma_] = syns
            if i.pos_ == 'VERB':
                patterns.append(
                    {'left': i.lemma_, 'right': i.right_edge.lemma_})
                if not i.right_edge.left_edge == i.right_edge:
                    patterns.append(
                        {'left': i.lemma_, 'right': i.right_edge.left_edge.lemma_})
                    patterns.append(
                        {'right': i.lemma_, 'left': i.right_edge.left_edge.lemma_})
                prev = i
                while True:
                    core = prev.head
                    print prev, core
                    if core.pos_ == 'NOUN':
                        patterns.append(
                            {'right': i.lemma_, 'left': core.lemma_})
                        patterns.append(
                            {'left': i.lemma_, 'right': core.lemma_})
                        break
                    elif core == prev:
                        break
                    prev = core
    return word_dump, patterns
    # if not i.is_stop and i.pos in [91, 99]:
    #     print get_words(i,30)


all_pos = pd.Series()
dump, patterns = get_wdump(desc)
for p in patterns:
    print p
    for x in dump[p['left']]:
        for y in dump[p['right']]:
            all_pos = all_pos.append(bridge(x, y, reflexive=False)[:5])
all_pos.shape
