from dfs_semantics import *
import sys
import nltk.sem.logic as fol

# World
world = MeaningSpace(file='model.observations')

fol_parser = fol.Expression.fromstring

# Lexicon
mike = world.getSemantics('mike')
jess = world.getSemantics('jess')
read = lambda x : merge(world.getSemantics('read'), x)
tease = lambda x: lambda y: merge(merge(world.getSemantics(world, 'tease'), x), y)



assert len(read(mike)) == 1, "Propositional meaning set is not singleton"
assert list(read(mike))[0].vec == list(world.getSemantics('read(mike)'))[0].vec, "Lambda derivation does not match full propositional meaning"


#u = ['mike', 'jess', 'winston']
someone_reads = world.exist(world.prop_dict[PARSE('tease(jess,mike)')], 'jess')
print(someone_reads)
#world.prop_dict[someone_reads.name] = someone_reads.vec
#someone_reads = world.exist(someone_reads, 'mike')
#print(someone_reads)
print(negation(implication(someone_reads, world.prop_dict[PARSE('tease(mike,jess)')])))
#print(negation(someone_reads))

p = list(world.propositions)[0]
q = list(world.propositions)[1]

pq = implication(conjunction(p, negation(q)), conjunction(p, q))

test = world.infer_meaningvec(PARSE('((read(jess) & sleep(mike)) | sleep(winston)) -> (tease(jess, mike) <-> tease(mike, jess))'))
print(test)
print(test.vec)

print(entails(test, test))
#print(world.exist(world.exist(test, 'jess'), 'mike'))
#print(world.exist(test, 'jess').vec)

#print(pq)
#print(pq.name.constants())
#print(pq.name.predicates())
#print(type(pq.name))


#for key, value in world.universe.items():
#    print(f"{key} : \n {value} \n")
