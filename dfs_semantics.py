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
import string
"""
 This file contains the base classes for the DFS datatypes:
    - MeaningVec (Meaning Vectors representing propositions)
    - MeaningSpace (The DFS meaning space)
    - MeaningSet (Sets representing subpropositional meaning)
 And some additional functional functions
"""

#Parsing first order logic.
PARSE = fol.Expression.fromstring
SYM = fol.Tokens()

def set_average(Set: set) -> np.array:
    matrix = set_to_matrix(Set)
    return MeaningVec(np.sum(matrix, axis=1, keepdims=True) / len(Set), name=None)

def marix_to_set(Matrix: np.array) -> set:
    return set(map(tuple, Matrix.T))

def set_to_matrix(Set):
    array = np.array([elem for elem in Set])
    return array.T

def smart_replace(formula):
    alphabet_list = list(string.ascii_lowercase)
    reg = re.search(rf'((?:z(?:[0-9]*)(?: ))*(?:z(?:[0-9]*))\.)', formula)
    if reg:
        m = ' '.join(reg.groups(0)).strip('.').split(' ')
        newvars = [alphabet_list[23+i] for i in range(len(m))]
        for i in range(len(m)):
            formula = formula.replace(m[i], newvars[i])
    return formula


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

def find_predicate(ex: str, var: fol.Variable) -> str:
    match = re.search(rf'([a-z]+)(\((?:[a-z]+,)*{str(var)}(?:,[a-z]+)*\))', ex)
    return match[1], match[2]

### Meaning Vector Object ###
class MeaningVec:
    def __init__(self, values: np.array, name=None):
        self.vec = np.array(values)
        #self.tup = tuple(self.vec)
        if isinstance(name, str):
            self.name = PARSE(name)
        elif isinstance(name, fol.Expression):
            self.name = name
        self.prob = np.sum(self.vec) / len(self.vec)

    def __len__(self):
        return len(self.vec)

    def __eq__(self, other):
        if isinstance(other, MeaningVec):
            return (self.vec == other.vec).all()
        return False

    def __hash__(self):
        return hash(repr(self))

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
def prob(a: MeaningVec) -> float:
    #double of the attribute
    return a.prob

def conditional_prob(a: MeaningVec, b: MeaningVec) -> float:
    return conjunction(a, b).prob / b.prob

def inference(a: MeaningVec, b: MeaningVec) -> float:
    if conditional_prob(a, b) > a.prob:
        return (conditional_prob(a, b) - a.prob) / (1 - a.prob)
    else:
        return (conditional_prob(a, b) - a.prob) / a.prob

###MeaningSet Object###
class MeaningSet:
    def __init__(self, denotation: str, world, vecs=None):
        self.denotation = denotation
        self.world = world
        self.closure = PARSE(str(self.denotation.simplify()).replace(f'{SYM.LAMBDA}', f'{SYM.EXISTS} '))
        if vecs:
            self.Set = set(vecs)
        else:
            self.Set = None

    def __call__(self, other):
        if isinstance(other, fol.Expression):
            reduction = self.denotation(other)
        else:
            reduction = self.denotation(other.denotation)
        return MeaningSet(reduction, self.world)

    def __len__(self):
        return len(self.set)

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.denotation}>"

    def __str__(self):
        return f"{smart_replace(str(self.denotation))}"

    def close(self):
        return self.world.infer_meaningset(self.closure)

    def simplify(self):
        return MeaningSet(self.denotation.simplify(), self.world, self.Set)

    def real(self):
        return set_average(self.world.infer_meaningset(self.closure))

class UnaryOperator(MeaningSet):
    def __init__(self, denotation: str, world, func, vecs=None):
        self.denotation = denotation
        self.world = world
        self.func = func
        if vecs:
            self.Set = set(vecs)
        else:
            self.Set = None
        super().__init__(denotation, world, vecs)

    def __call__(self, other):
        if isinstance(other, fol.Expression):
            reduction = self.denotation(other)
        else:
            reduction = self.denotation(other.denotation)
        return UnaryOperator(reduction, self.world, self.func, self.Set)

    def simplify(self):
        return UnaryOperator(self.denotation.simplify(), self.world, self.func, self.Set)

    def close(self):
        print(self.denotation.negate())
        try:
            content = self.world.infer_meaningset(self.denotation.negate())
            return self.func(content)
        except:
            print("Higher order closure not implemented yet.")

### Meaning Space object ###
class MeaningSpace:
    def __init__(self, file=None, propositions=None):
        if file:
            self.matrix, self.propositions, self.prop_list = self.readSpace(file) #matrix, list of MeaningVecs, list of tuples and strings

        elif propositions:
            self.matrix, self.propositions, self.prop_list = self.canonicalSpace(propositions)

        self.prop_dict = {str(prop.name) : prop for prop in self.propositions} #from fol expressions to MeaningVectors, TODO: I add all vectors that are computed here for memoization
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
                #NLTK returns sets of variables, which are unordered. Order matters, so I parse the raw strings to infer the arguments
                re_entities = re.search(rf'(?:\()[a-z]+(,[a-z]+)*(?:\))', str(prop.name))
                local_entities = tuple([fol.Variable(elem) for elem in re_entities.group(0).strip('(').strip(')').split(',')])
                assert prop.name.constants() == set(local_entities) # emergency check that parsing works
                universe[local_predicate.name].add(local_entities)
            else:
                universe[local_predicate.name].update(local_entities)
        return universe

    def generate_dummy(self, ex: fol.Expression, var: fol.Variable) -> fol.Variable:
        """ Pick a dummy variable that is consistent with the interpretation function of the predicate to which the argument belongs.
        """
        relevant_pred, arguments = find_predicate(str(ex), var)
        local_entities = tuple([fol.Variable(elem) for elem in arguments.strip('(').strip(')').split(',')])
        argarity = local_entities.index(var)
        if argarity > 0:
            possible_args = set([elem[argarity] for elem in list(self.universe[relevant_pred])])
            dummy = list(possible_args)[0]
        else:
            dummy = list(self.universe[relevant_pred])[0][argarity]

        return ex.replace(var, PARSE(dummy.name)), dummy

    def is_atomic(self, expression: fol.Expression) -> bool:
        if expression in [v.name for v in self.propositions]:
            return True
        else:
            return False

    def is_defined(self, expression: fol.Expression) -> bool:
        terms = list(expression.constants().union(expression.variables()))
        predicate, arguments = find_predicate(str(expression), terms[0])
        arguments = tuple([fol.Variable(item) for item in arguments.strip('(').strip(')').split(',')])
        sys.exit()

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

    def modelEntailment(self, MeaningVec: MeaningVec) -> List[MeaningVec]:
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
        str_expression = str(expression)
        self._calls += 1

        #Stop condition: we arrive at an atomic proposition
        if self.is_atomic(expression):
            return self.prop_dict[str_expression]
        #If the meaning of an expression is not defined (contains free variables or unknown constants), return zero vector or raise an error
        elif not self.is_atomic(expression) and isinstance(expression, fol.ApplicationExpression):
            possible_args = list(expression.constants().union(expression.variables()))
            if find_predicate(str(expression), possible_args[0])[0] not in self.universe.keys():
                raise KeyError('predicate not defined in universe')
            return MeaningVec(list(np.zeros(self.matrix.shape[0], dtype=int)), name=expression)

        #this function is memoized with the use of self.prop_dict
        if str_expression not in self.prop_dict:
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
                #introduce dummy variable, since it will be existentially closed anyway, always choose a unique dummy
                dummy = [e for e in list(self.universe['entities']) if e not in term.constants()][0]
                term = term.replace(variable, PARSE(dummy.name))

        #the recursive step
            if operator == f'{SYM.AND}':
                self.prop_dict[str_expression] = conjunction(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.OR}':
                self.prop_dict[str_expression] = disjunction(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.IMP}':
                self.prop_dict[str_expression] = implication(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.IFF}':
                self.prop_dict[str_expression] = equivalence(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.EXISTS}':
                self.prop_dict[str_expression] = self.exist(self.infer_meaningvec(term),  dummy.name)
            elif operator == f'{SYM.ALL}':
                self.prop_dict[str_expression] = self.all(self.infer_meaningvec(term), dummy.name)

        return self.prop_dict[str_expression]

    def assignment_function(self, phi: MeaningVec, c: fol.Variable) -> List[MeaningVec]:
        """
        Returns the list of Meaning Vectors of proposition phi according to the assignment function g[c/x].
        Only replaces with entities in the interpretation function of phi
        List: [phi_g[c/e1], phi_g[c/e2], ... , phi_g[c/en]]
        """

        entity_names = [e.name for e in self.universe['entities']]
        predicates = [phi.name.replace(c, PARSE(e)) for e in entity_names]
        #TODO: quick and dirty
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
        Get the MeaningSet for a word in the meaning space using disjunctive semantics
        """
        relevantProps = [MeaningVec(vector, name=prop) for vector, prop in self.prop_list if word in prop]
        return set(relevantProps)

    def infer_meaningset(self, expression: fol.Expression) -> MeaningSet:
        vectors = []
        meaningvec = self.infer_meaningvec(expression)
        for p in self.propositions:
            if entails(p, meaningvec):
                vectors.append(p)
        return set(vectors)
