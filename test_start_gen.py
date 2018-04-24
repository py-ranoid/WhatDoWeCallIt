from portmanteau import bridge
from pandas import Series
from DescProc.start import get_wdump, get_keywords

MIN_SIZE = 6

description = raw_input("Enter a description of your product :")
description = "A website to rent cameras" if not description else description

dump, patterns = get_wdump(description)
using = set()

print "****"
for i, p in enumerate(patterns):
    print str(i) + ". " + p['left'] + ' + ' + p['right']

choices = raw_input(
    'Which patterns would you like to drop? \n(Numbers seperated by spaces, Enter if all patterns are relevant)\n :')
drop_indices = set(map(int, choices.strip().split()))

pattern_pool = []
for i in range(len(patterns)):
    try:
        print i in drop_indices
        if i not in drop_indices:
            pattern_pool.append(patterns[i])
        print pattern_pool
    except:
        pass

for p in pattern_pool:
    using.update({p['left'], p['right']})

for w in using:
    dump[w] = get_keywords(w, dump)

all_pos = Series()
for p in pattern_pool:
    print p
    for x in dump[p['left']]:
        for y in dump[p['right']]:
            all_pos = all_pos.append(bridge(x, y, reflexive=False)[:5])
print ("\nPrinting upto 15 names :")
print all_pos[all_pos.index.str.len() > MIN_SIZE].sort_values(ascending=False)[:15]
