from dfs_semantics import *
import pandas as pd
from itertools import product
from nltk import Tree
import numpy as np
from functools import reduce


# World
world = MeaningSpace(file='worlds/wollic.observations')

#operators
AND = MeaningSet(PARSE('\\P \\Q \\x. (P(x) & Q(x))'), world)

#predicates
PAY = MeaningSet(PARSE('\\x. (pay(x))'), world)
LEAVE = MeaningSet(PARSE('\\x. (leave(x))'), world)
DRINK = MeaningSet(PARSE('\\y \\x. (drink(x, y))'), world)
EAT = MeaningSet(PARSE('\\y \\x. (eat(x, y))'), world)
ENTER = MeaningSet(PARSE('\\y \\x. (enter(x, y))'), world)
ORDER = MeaningSet(PARSE('\\y \\x. (order(x, y))'), world)
ASKMENU = MeaningSet(PARSE('\\x. (askmenu(x))'), world)

predicates = [PAY, LEAVE, DRINK, EAT, ENTER, ORDER]

# entities
ellen = PARSE('ellen')
john = PARSE('john')
beer = PARSE('beer')
wine = PARSE('wine')
restaurant = PARSE('restaurant')
bar = PARSE('bar')
pizza = PARSE('pizza')
fries = PARSE('fries')
entities = [ellen, john, beer, wine, restaurant, bar, pizza, fries]

#everything
all = [ellen, john, beer, wine, restaurant, bar, pizza, fries, PAY, LEAVE, DRINK, EAT, ENTER, ORDER]

def cosine_similarity(a, b):
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def setnegate(a: Set[MeaningVec]) -> Set:
    return set([negation(prop) for prop in a])

prop_cosine = []
for vector in world.propositions:
    p = vector.vec
    neg_p = negation(vector).vec
    cosine = cosine_similarity(p, neg_p)
    prop_cosine.append(cosine)

exist_cosine = []
for predicate in predicates:
    p = world.infer_meaningvec(predicate.closure)
    neg_p = negation(p)
    exist_cosine.append(cosine_similarity(p.vec, neg_p.vec))

real_cosine = []
for predicate in predicates:
    p = predicate.real()
    neg_p = set_average(setnegate(predicate.close())).vec.T
    real_cosine.append(cosine_similarity(p, neg_p))

print("Propositional negation: ", prop_cosine)
print("Exitential closure negation: ", exist_cosine)
print("Set average negation: ", real_cosine)

for predicate in predicates[-1:-1]:
    vectors = predicate.close()
    neg_vectors = setnegate(predicate.close())
    vectors_avg = reduce(np.add, vectors) / len(vectors)
    neg_vectors_avg = reduce(np.add, neg_vectors) / len(neg_vectors)
    print(predicate, ' '*(16-len(str(predicate))), np.round(cosine_similarity(vectors_avg, neg_vectors_avg), 4), np.mean(vectors_avg+neg_vectors_avg))


vectors = ORDER.close()
neg_vectors = setnegate(ORDER.close())
vectors_avg = reduce(np.add, vectors) / len(vectors)
neg_vectors_avg = reduce(np.add, neg_vectors) / len(neg_vectors)
for vec in neg_vectors:
    print(vec, ' '*(25-len(str(vec))), vec.prob)
print('=============================================================')
for vec in vectors:
    print(vec, ' '*(25-len(str(vec))), vec.prob)

print(prob(neg_vectors_avg))
