import csv
import numpy as np
import pandas as pd
from itertools import product
from typing import *
from functools import reduce
import nltk.sem.logic as fol
from collections import defaultdict

#Parsing first order logic.
PARSE = fol.Expression.fromstring
SYM = fol.Tokens()

### Practical ###
def marix_to_set(Matrix):
    return set(map(tuple, Matrix.T))

def set_to_matrix(Set):
    array = np.array([elem for elem in Set])
    return array.T

### Meaning Vector Object ###
class MeaningVec:
    def __init__(self, values: List[int], name=None):
        self.vec = tuple(values)
        self.name = PARSE(name)
        self.prob = sum(self) / len(self)

    def __len__(self):
        return len(self.vec)

    def __getitem__(self, index):
        return self.vec[index]

    def __repr__(self):
        return f"{self.name}"

MeaningSet = NewType("MeaningSet", Set[MeaningVec]) # set of meaning vectors

### Propositional Logic ###
def negation(a: MeaningVec) -> MeaningVec:
    if a.name:
        newname = f'{SYM.NOT}{a.name}'
    else:
        newname = None
    return MeaningVec([int(not i) for i in a], name=newname)

def conjunction(a: MeaningVec, b: MeaningVec) -> MeaningVec:
    if a.name and b.name:
        newname = f'{a.name} {SYM.AND} {b.name}'
    else:
        newname = None
    return MeaningVec([i*j for i, j in zip(a, b)], name=newname)

def disjunction(a: MeaningVec, b: MeaningVec) -> MeaningVec:
    if a.name and b.name:
        newname = f'{a.name} {SYM.OR} {b.name}'
    else:
        newname = None
    return MeaningVec(negation(conjunction(negation(a), negation(b))).vec, name=newname)

def implication(a: MeaningVec, b: MeaningVec) -> MeaningVec:
    if a.name and b.name:
        newname = f'{a.name} {SYM.IMP} {b.name}'
    else:
        newname = None
    return MeaningVec(disjunction(negation(a), b), name=newname)

def equivalence(a: MeaningVec, b: MeaningVec) -> MeaningVec:
    if a.name and b.name:
        newname = f'{a.name} {SYM.IFF} {b.name}'
    else:
        newname = None
    return MeaningVec(conjunction(implication(a, b), implication(b, a)), name=newname)

def entails(a: MeaningVec, b: MeaningVec) -> bool:
    """ Returns True if the the vector implication of a --> b contains only True (ie. entailment) and False otherwise
    """
    return int(True) if all(t == True for t in implication(a, b)) else int(False)

### Probability theory ###
def prob(a: MeaningVec) -> float:
    #double of the attribute
    return a.prob

def conditional_prob(a: MeaningVec, b: MeaningVec) -> float:
    return conjunction(a, b).prob / a.prob

def inference(a: MeaningVec, b: MeaningVec) -> float:
    if conditional_prob(a, b) > a.prob:
        return (conditional_prob(a, b) - a.prob) / (1 - a.prob)
    else:
        return (conditional_prob(a, b) - a.prob) / a.prob

### Meaning Space object ###
class MeaningSpace:
    def __init__(self, file=None, propositions=None):
        if file:
            self.matrix, self.propositions, self.prop_list = self.readSpace(file) #matrix, list of MeaningVecs, list of tuples and strings

        elif propositions:
            self.matrix, self.propositions, self.prop_list = self.canonicalSpace(propositions)

        self.prop_dict = {prop.name : prop for prop in self.propositions} #from fol expressions to MeaningVectors, TODO: I add all vectors that are computed here for memoization
        self.dim = self.matrix.shape
        self.universe = self.find_universe() #dictionary of entities and relations

    def __len__(self):
        return len(self.propositions)

    def __str__(self):
        return f"{set([self.prop_list[elem][1] for elem in matrix_to_set(self.matrix)])}"

    def find_universe(self):
        universe = defaultdict(lambda : set())
        for prop in self.propositions:
            assert prop.name.is_atom(), "Only atomic propositions allowed in model definition"
            local_entities = prop.name.constants()
            local_predicate = list(prop.name.predicates())[0] #if only atomic propositions, then len of this HAS to be 1
            universe['entities'].update(local_entities)
            if len(local_entities) > 1:
                universe[local_predicate.name].add(tuple(local_entities))
            else:
                universe[local_predicate.name].update(local_entities)

        return universe

    def is_atomic(self, expression: fol.Expression) -> bool:
        if expression in self.propositions:
            return True

    def readSpace(self, file):
        model_df = pd.read_csv(file, sep=' ', header=0)
        model_matrix = np.array(model_df)
        names = list(model_df.columns)
        prop_list = [(tuple(np.array(model_df.iloc[:, index])), prop) for index, prop in enumerate(names)]
        Set = set([MeaningVec(np.array(model_df.iloc[:, index]), name=prop) for index, prop in enumerate(names)])
        return model_matrix, Set, prop_list

    def canonicalSpace(self, props):
        space = [list(i) for i in product([0,1], repeat=len(props))]

        model_df = pd.DataFrame.from_records(space)
        names = props

        model_matrix = np.array(model_df)
        prop_list = [(tuple(np.array(model_df.iloc[:, index])), prop) for index, prop in enumerate(names)]
        Set = set([MeaningVec(np.array(model_df.iloc[:, index]), name=prop) for index, prop in enumerate(names)])

        return model_matrix, Set, prop_list

    def printSet(self, Set):
        try:
            print(set([self.prop_list[elem][1] for elem in Set]))
        except KeyError:
            print(set([f'v_{i}' for i, elem in enumerate(Set)]))

    def modelEntailment(self, MeaningVec):
        assert len(MeaningVec) == self.dim[0]
        entailments = []
        for p in self.propositions:
            if entails(p, MeaningVec):
                entailments.append(p)
        return entailments

    def infer_meaningvec(self, expression: fol.Expression) -> MeaningVec:
        #Stop condition: we arrive at an atomic proposition
        if self.is_atomic(expression):
            return self.prop_dict[expression]

        #this function is memoized with the use of self.prop_dict
        if expression not in self.prop_dict:
            #NLTK has no functionality for returning terms of a formula, so I do it a ghetto way
            try: #see if its a logical connective
                operator = expression.getOp()
                split = str(expression).split(f' {operator} ')
                assert len(split) == 2
                first_term = PARSE(split[0][1:]) #fix parenthesis
                second_term = PARSE(split[1][:-1])
            except AttributeError: #otherwise maybe a quantifier
                operator = expression.getQuantifier()
                variable = expression.variable
                term = PARSE(str(expression).split(f'{operator} {variable}.')[-1])

            if operator == f'{SYM.AND}':
                self.prop_dict[expression] = conjunction(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.OR}':
                self.prop_dict[expression] = disjunction(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.IMP}':
                self.prop_dict[expression] = implication(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.IFF}':
                self.prop_dict[expression] = equivalence(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.EXISTS}': ###ONLY WORKS FOR 1 QUANTIFIER TODO: reconstruct using regex
                self.prop_dict[expression] = self.exist(self.infer_meaningvec(term), variable)


        return self.prop_dict[expression]

    #only works 1 embedding deep now
    def exist(self, expression: MeaningVec, var: str) -> MeaningVec:
        #detect to be bound constant/free variable, pick a new variable name
        to_be_bound = fol.Variable(var)
        newvar = fol.Variable('z1')
        if newvar in expression.name.variables() or expression.name.free():
            newvar = fol.unique_variable(pattern=newvar)

        #make sure variable to be bound is valid
        assert expression.name is not None
        assert to_be_bound in expression.name.constants(), "Variable does not occur in expression"

        #create a new assignment function (TODO: ghetto, NLTK has functionality for this)
        #look up all the vectors associated with new assignment (TODO: Only works on atomic props)
        entity_names = [e.name for e in self.universe['entities']]
        predicates = [expression.name.replace(to_be_bound, PARSE(e)) for e in entity_names]
        vectors = [self.infer_meaningvec(pred) for pred in predicates]

        # Take disjunction over assignment
        exist = reduce(disjunction, vectors)
        newname = PARSE(f'{SYM.EXISTS} {newvar}. {expression.name}')
        exist.name = newname.replace(to_be_bound, PARSE(newvar.name), alpha_convert=False)
        return exist

    def getSemantics(self, word: str) -> MeaningSet:
        relevantProps = [MeaningVec(vector, name=prop) for vector, prop in self.prop_list if word in prop]
        return set(relevantProps)

def merge(predicate: MeaningSet, argument: MeaningSet) -> MeaningSet:
    merged_meaning = []
    for argument_p in argument:
        for functor_p in predicate:
            if entails(argument_p, functor_p):
                merged_meaning.append(argument_p)
                break
    return set(merged_meaning)
