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
