import csv
import numpy as np
import pandas as pd
from itertools import product
from typing import *
from functools import reduce
import nltk.sem.logic as fol
from collections import defaultdict
import sys
import re

#Parsing first order logic.
PARSE = fol.Expression.fromstring
SYM = fol.Tokens()

### Practical ###
def set_average(set):
    matrix = set_to_matrix(set)
    return np.sum(matrix, axis=1, keepdims=True) / len(set)

def marix_to_set(Matrix):
    return set(map(tuple, Matrix.T))

def set_to_matrix(Set):
    array = np.array([elem for elem in Set])
    return array.T

def deconstruct_expression(expression: str) -> str:
    """
    Function to remove the outermost quantifier in a string formatted like an nltk.sem.logic.Expression
    """
    match = re.match(rf'({SYM.EXISTS}|{SYM.ALL}) (([a-z][0-9]*)(?: ))*(([a-z][0-9]*)(?:\.))', expression)
    variablebinder = match[0].strip('.').split(' ')
    quantifier = variablebinder[0]
    variables = variablebinder[1:]

    if len(variables) > 1: #if there are more than 1 variable, we need to only delete the first one
        new_variables = variables[1:]
        new_quantifier = quantifier + ' ' + ' '.join(new_variables) #this is bad, but what do?
        expression = expression.strip(f'{variablebinder}')
        return new_quantifier + expression

    else: #if theres only one variable, we remove the whole quantifier
        remove = ' '.join(variablebinder) + '.'
        return expression.split(f'{remove}')[-1]

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

### Meaning Set Object
class MeaningSet:
    def __init__(self, word, world, name=None):
        self.set = world.getSemantics(word)
        self.name = PARSE(name)
        self.real = set_average(self.set)

    def __len__(self):
        return len(self.set)

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
        self._calls=0

    def __len__(self):
        return len(self.propositions)

    def __str__(self):
        return f"{set([self.prop_list[elem][1] for elem in matrix_to_set(self.matrix)])}"

    def find_universe(self):
        """
        Construct the universe for this meaning space. The entities and the predicates (and their possible inputs)
        """
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

    def prettyprint(self):
        rownames = [f"M{i}" for i, row in enumerate(self.matrix)]
        columnnames = [vector.name for vector in list(self.propositions)]
        df = pd.DataFrame(self.matrix, index=rownames, columns=columnnames)
        print(df)

    def modelEntailment(self, MeaningVec: MeaningVec):
        """
        Returns all atomic propositions in the meaning spaces that entail MeaningVec
        """
        assert len(MeaningVec) == self.dim[0]
        entailments = []
        for p in self.propositions:
            if entails(p, MeaningVec):
                entailments.append(p)
        return entailments

    def infer_meaningvec(self, expression: fol.Expression) -> MeaningVec:
        """
        Recursively infers the meaning vector in the meaning space for an arbitrary first order logic expression, granted the syntax is correct.
        """
        self._calls += 1
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
                term = PARSE(deconstruct_expression(str(expression)))
                #a predicate with a free variable has no meaning
                #introduce dummy variable, since it will be existentially closed anyway
                #always choose a unique dummy
                dummy = [e for e in list(self.universe['entities']) if e not in term.constants()][0]
                term = term.replace(variable, PARSE(dummy.name))

            if operator == f'{SYM.AND}':
                self.prop_dict[expression] = conjunction(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.OR}':
                self.prop_dict[expression] = disjunction(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.IMP}':
                self.prop_dict[expression] = implication(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.IFF}':
                self.prop_dict[expression] = equivalence(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.EXISTS}':
                self.prop_dict[expression] = self.exist(self.infer_meaningvec(term),  dummy.name)
            elif operator == f'{SYM.ALL}':
                self.prop_dict[expression] = self.all(self.infer_meaningvec(term), dummy.name)

        return self.prop_dict[expression]

    def assignment_function(self, phi: MeaningVec, c: fol.Variable) -> List[MeaningVec]:
        """
        Returns the list of Meaning Vectors of proposition phi according to the assignment function g[c/x]
        List: [phi_g[c/e1], phi_g[c/e2], ... , phi_g[c/en]]
        """
        entity_names = entity_names = [e.name for e in self.universe['entities']]
        predicates = [phi.name.replace(c, PARSE(e)) for e in entity_names]
        return [self.infer_meaningvec(pred) for pred in predicates]

    def exist(self, expression: MeaningVec, var: str) -> MeaningVec:
        """
        Existentially binds a propositional meaning vector for some variable.
        """
        #detect to be bound constant/free variable, pick a new variable name
        to_be_bound = fol.Variable(var)
        newvar = fol.unique_variable()

        #make sure variable to be bound is valid
        assert expression.name is not None
        assert to_be_bound in expression.name.constants(), "Variable does not occur in expression"

        #look up all the vectors associated with new assignment
        vectors = self.assignment_function(expression, to_be_bound)

        # Take disjunction over assignment
        exist = reduce(disjunction, vectors)
        newname = PARSE(f'{SYM.EXISTS} {newvar}. ({expression.name})')
        exist.name = newname.replace(to_be_bound, PARSE(newvar.name), alpha_convert=False)
        return exist

    def all(self, expression: MeaningVec, var: str) -> MeaningVec:
        """
        Universally binds a propostional meaning vector for some variable.
        """
        #detect to be bound constant/free variable, pick a new variable name
        to_be_bound = fol.Variable(var)
        newvar = fol.unique_variable()

        #make sure variable to be bound is valid
        assert expression.name is not None
        assert to_be_bound in expression.name.constants(), "Variable does not occur in expression"

        #look up all the vectors associated with new assignment
        vectors = self.assignment_function(expression, to_be_bound)

        # Take conjunction over assignment
        all = reduce(conjunction, vectors)
        newname = PARSE(f'{SYM.ALL} {newvar}. ({expression.name})')
        all.name = newname.replace(to_be_bound, PARSE(newvar.name), alpha_convert=False)
        return all

    def getSemantics(self, word: str) -> MeaningSet:
        """
        Get the MeaningSet for a word in the meaning space
        """
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
