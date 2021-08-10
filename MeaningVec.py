import csv
import numpy as np
import pandas as pd
from itertools import product
from typing import *
from functools import reduce
import nltk.sem.logic as fol
from collections import defaultdict
import sys
from parsing import *
import statistics

"""
This file contains functionality for Meaning vectors and vector logic
- MeaningVec object
- Propositional logic
- Probability theory
"""

### Meaning Vector Object ###
class MeaningVec:
    def __init__(self, values: np.array, name=None):
        self.vec = np.array(values)
        #self.tup = tuple(self.vec)
        if isinstance(name, str):
            self.name = PARSE(name)
        else:
            self.name = name
        self.prob = np.sum(self.vec) / len(self.vec)

    def __len__(self):
        return len(self.vec)

    def __eq__(self, other):
        if isinstance(other, MeaningVec):
            return (self.vec == other.vec).all()
        return False

    def __hash__(self):
        return hash(tuple(self.vec))

    def __getitem__(self, index):
        return self.vec[index] #list lookup is faster than array lookup!

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.name}>"

    def __str__(self):
        return f"{smart_replace(str(self.name))}"

### Propositional Logic ###
def negation(a: MeaningVec) -> MeaningVec:
    if a.name:
        newname = f'{SYM.NOT}{a.name}'
    else:
        newname = None
    #return MeaningVec([int(not i) for i in a], name=newname)
    return MeaningVec(1-a.vec, name=newname)

def conjunction(a: MeaningVec, b: MeaningVec) -> MeaningVec:
    if a.name and b.name:
        newname = f'{a.name} {SYM.AND} {b.name}'
    else:
        newname = None
    #return MeaningVec([i*j for i, j in zip(a, b)], name=newname)
    return MeaningVec(a.vec*b.vec, name=newname)

def disjunction(a: MeaningVec, b: MeaningVec) -> MeaningVec:
    if a.name and b.name:
        newname = f'{a.name} {SYM.OR} {b.name}'
    else:
        newname = None
    #return MeaningVec(negation(conjunction(negation(a), negation(b))).vec, name=newname)
    return MeaningVec(np.logical_or(a.vec, b.vec), name=newname) #numpy is faster than the above


def implication(a: MeaningVec, b: MeaningVec) -> MeaningVec:
    if a.name and b.name:
        newname = f'{a.name} {SYM.IMP} {b.name}'
    else:
        newname = None
    return MeaningVec(disjunction(negation(a), b).vec, name=newname)

def equivalence(a: MeaningVec, b: MeaningVec) -> MeaningVec:
    if a.name and b.name:
        newname = f'{a.name} {SYM.IFF} {b.name}'
    else:
        newname = None
    #return MeaningVec(conjunction(implication(a, b), implication(b, a)).vec, name=newname)
    return MeaningVec(np.equal(a.vec, b.vec), name=newname)

def entails(a: MeaningVec, b: MeaningVec) -> bool:
    """ Returns True if the the vector implication of a --> b contains only True (ie. entailment) and False otherwise
    """
    #return int(True) if all(t == True for t in implication(a, b)) else int(False)
    return np.all(implication(a, b).vec)

### Probability theory ###
def prob(a: np.array) -> float:
    return sum(a)/len(a)

def conditional_prob(a: MeaningVec, b: MeaningVec) -> float:
    """ conditional_prob(a, b) = P(A|B) """
    return conjunction(a, b).prob / b.prob

def conditional_prob_raw(a: np.array, b: np.array) -> float:
    """ conditional_prob(a, b) = P(A|B) """
    return prob(a*b) / prob(b)

def entropy(a: tuple) -> float:
    if len(a) == 2:
        return -1 * np.log2(conditional_prob(a[0], a[1]))
    else:
        return -1 * np.log2(a.prob)

def conditional_entropy(tup: tuple) -> float:
    return -1 * np.log2(conditional_prob(tup[0], tup[1]))

def conditional_set_entropy(a: set, b: set) -> float:
    return statistics.mean(map(entropy, product(a, b)))

def inference(a: MeaningVec, b: MeaningVec) -> float:
    if conditional_prob(a, b) > a.prob:
        return (conditional_prob(a, b) - a.prob) / (1 - a.prob)
    else:
        return (conditional_prob(a, b) - a.prob) / a.prob

def inference_raw(a: np.array, b: np.array) -> float:
    posterior = conditional_prob_raw(a, b)
    prior = prob(a)
    if posterior > prior:
        return (posterior - prior) / (1 - prior)
    else:
        return (posterior - prior) / prior
