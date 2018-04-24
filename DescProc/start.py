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
from pandas import DataFrame
# from portmanteau import bridge

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
        return DataFrame(words).T.sort_values('sim', ascending=False)


def get_words(w, lim=10):
    result = get_df(w)
    print result
    return None if result is None else result[:lim].index


def get_wdump(desc="A friend for paying on websites."):
    word_dump = {}
    patterns = []
    for i in nlp(unicode(desc)):
        print i
        if not i.is_stop and i.pos_ in ['ADJ', 'NOUN', 'VERB']:
            syns = list(get_words(i, 9))
            syns = syns[:-1] + [i.lemma_] if i.lemma_ not in syns else syns
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


def get_keywords(word, dump):
    print "*******\nWord :", word
    enum = enumerate(dump[word])
    formatted = [(str(index) + '. ' + dw).ljust(16) for index, dw in enum]
    print ''.join(formatted[:5])
    print ''.join(formatted[5:])
    choices = raw_input(
        'Which synonyms would you like to drop? \n(Numbers seperated by spaces, Enter if all words are relevant)\n :')
    word_pool = set()
    while True:
        more = raw_input('Enter a synonym for "' +
                         word + '" (Enter to pass) :')
        if not more:
            break
        else:
            word_pool.add(more.strip())
    drop_indices = set(map(int, choices.strip().split()))
    for i in range(len(dump[word])):
        try:
            if i not in drop_indices:
                word_pool.add(dump[word][i])
        except:
            pass
    return word_pool
