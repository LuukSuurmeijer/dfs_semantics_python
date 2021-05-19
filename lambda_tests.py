from dfs_semantics import *
import sys
import nltk.sem.logic as fol
import re

# World
world = MeaningSpace(file='model.observations')

fol_parser = fol.Expression.fromstring

# Lexicon
#mike = world.getSemantics('mike')
#jess = world.getSemantics('jess')
#read = lambda x : merge(world.getSemantics('read'), x)
#tease = lambda x: lambda y: merge(merge(world.getSemantics(world, 'tease'), x), y)

#assert len(read(mike)) == 1, "Propositional meaning set is not singleton"
#assert list(read(mike))[0].vec == list(world.getSemantics('read(mike)'))[0].vec, "Lambda derivation does not match full propositional meaning"


#u = ['mike', 'jess', 'winston']
#someone_reads = world.exist(world.prop_dict[PARSE('tease(jess,mike)')], 'jess')
#print(someone_reads)
#print(negation(implication(someone_reads, world.prop_dict[PARSE('tease(mike,jess)')])))

#p = list(world.propositions)[0]
#q = list(world.propositions)[1]

#pq = implication(conjunction(p, negation(q)), conjunction(p, q))

test = world.infer_meaningvec(PARSE('read(jess) & sleep(mike)'))
print(test)
print(test.vec)
all = world.all(test, 'mike')
print(all)
print(all.vec)
world._calls = 0
all2 = world.all(all, 'jess')
print(all2)
print(all2.vec)
print(world._calls)

#world.prettyprint()

#assert entails(test, ex)
#assert entails(ex, ex2)
#assert entails(test, ex2)

#one = world.infer_meaningvec(PARSE('exist z1. (read(z1) & sleep(jess))'))
#print(world.calls)
#world.calls = 0
#two = world.infer_meaningvec(PARSE('exist z2. (exist z1. (read(z1) & sleep(z2)))'))
#print(world.calls)

#assert entails(one, two)
