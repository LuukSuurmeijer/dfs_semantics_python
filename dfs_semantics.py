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
from MeaningVec import *
"""
 This file contains the base classes for the DFS datatypes:
    - MeaningSpace (The DFS meaning space)
    - MeaningSet (Sets representing subpropositional meaning)
    - UnaryOperator, BinaryOperator and Quantifier
 And some additional functional functions
"""

def set_average(Set: set) -> np.array:
    if len(Set) > 1:
        return reduce(np.add, Set) / len(Set)
    elif len(Set) == 1:
        return list(Set)[0]
    else:
        return [0]

def marix_to_set(Matrix: np.array) -> set:
    return set(map(tuple, Matrix.T))

def set_to_matrix(Set):
    array = np.array([elem for elem in Set])
    return array.T

###MeaningSet Object###
class MeaningSet:
    def __init__(self, denotation: str, world):
        self.denotation = denotation
        self.world = world
        self.closure = string_close_lambda(self.denotation.simplify())

    def __call__(self, other):
        if isinstance(other, fol.Expression):
            reduction = self.denotation(other)
        else:
            reduction = self.denotation(other.denotation)
        return MeaningSet(reduction, self.world)

    def __len__(self):
        return len(self.close())

    def __repr__(self):
        return f"<{self.__class__.__name__} {self.denotation}>"

    def __str__(self):
        return f"{smart_replace(str(self.denotation))}"

    def close(self):
        return self.world.infer_meaningset(self.closure)

    def simplify(self):
        return MeaningSet(self.denotation.simplify(), self.world)

    def real(self):
        return set_average(self.close())

class UnaryOperator(MeaningSet):
    def __init__(self, denotation: fol.Expression, world, func):
        self.denotation = denotation
        self.world = world
        self.func = func

        self.closure = string_close_lambda(self.denotation.simplify())
        self.FOL_content = get_term(self.denotation.simplify())
        super().__init__(denotation, world)

    def __call__(self, other):
        if isinstance(other, fol.Expression):
            reduction = self.denotation(other)
        else:
            reduction = self.denotation(other.denotation)
        return UnaryOperator(reduction, self.world, self.func)

    def simplify(self):
        return UnaryOperator(self.denotation.simplify(), self.world, self.func)

    def close(self):
        set_content = self.world.infer_meaningset(self.closure)
        return self.func(set_content)

class BinaryOperator(MeaningSet):
    def __init__(self, denotation: fol.Expression, world, func):
        self.denotation = denotation
        self.world = world
        self.func = func

        self.FOL_content = get_term(self.denotation.simplify())
        super().__init__(denotation, world)

    def __call__(self, other):
        if isinstance(other, fol.Expression):
            #other = PARSE(flip(other))
            reduction = self.denotation(other)
        else:
            #other.denotation = PARSE(flip(other.denotation))
            reduction = self.denotation(other.denotation)
        return BinaryOperator(reduction, self.world, self.func)

    def simplify(self):
        return BinaryOperator(self.denotation.simplify(), self.world, self.func)

    def close(self):
        first = self.world.infer_meaningvec(string_close(self.FOL_content.first))
        second = self.world.infer_meaningvec(string_close(self.FOL_content.second))
        return self.func(self.world.infer_meaningset(first.name), self.world.infer_meaningset(second.name))

class Quantifier(MeaningSet):
    def __init__(self, denotation: fol.Expression, world, func):
        self.denotation = denotation
        self.world = world
        self.func = func
        self.FOL_content = get_term(self.denotation.simplify())
        super().__init__(denotation, world)

    def __call__(self, other):
        if isinstance(other, fol.Expression):
            #other = PARSE(flip(other))
            reduction = self.denotation(other)
        else:
            #other.denotation = PARSE(flip(other.denotation))
            reduction = self.denotation(other.denotation)
        return Quantifier(reduction, self.world, self.func)

    def simplify(self):
        return Quantifier(self.denotation.simplify(), self.world, self.func)

    def close(self):
        dummy = self.world.generate_dummy(self.FOL_content, self.denotation.variable)
        newterm = self.FOL_content.replace(self.denotation.variable, PARSE(str(dummy)))
        assignment = self.world.proto_assignment_function(self.world.infer_meaningvec(newterm), dummy)
        return set(reduce(self.func, [self.world.infer_meaningset(string_close(prop.name)) for prop in assignment]))

### Meaning Space object ###
class MeaningSpace:
    def __init__(self, file=None, propositions=None, signature=None):
        if file:
            self.matrix, self.propositions, self.prop_list = self.readSpace(file) #matrix, list of MeaningVecs, list of tuples and strings

        elif propositions:
            self.matrix, self.propositions, self.prop_list = self.canonicalSpace(propositions)

        self.type_signature = signature
        self.prop_dict = {str(prop.name) : prop for prop in self.propositions} #from fol expressions to MeaningVectors, TODO: I add all vectors that are computed here for memoization
        self.dim = self.matrix.shape
        self.universe = self.find_universe(sig=self.type_signature) #dictionary of entities and relations

        for entity in self.universe[T_PARSE('e')]:
            self.type_signature[str(entity)] = 'e'

        self._calls=0

    def __len__(self):
        return len(self.propositions)

    def __str__(self):
        return f"{set([self.prop_list[elem][1] for elem in matrix_to_set(self.matrix)])}"

    def find_universe(self, sig=None):
        universe = defaultdict(lambda : defaultdict(lambda : set()))
        universe[T_PARSE('e')] = set()
        for prop in self.propositions:
            assert prop.name.is_atom(), "Only atomic propositions allowed in model definition"
            local_entities = prop.name.constants()
            local_predicate = list(prop.name.predicates())[0] #if only atomic propositions, then len of this HAS to be 1
            typename = prop.name.typecheck(signature=sig)[str(local_predicate)]
            universe[T_PARSE('e')].update(local_entities)
            if not isinstance(typename, fol.EntityType):
                if len(local_entities) > 1:
                    #NLTK returns sets of variables, which are unordered. Order matters, so I parse the raw strings to infer the arguments
                    re_entities = re.search(rf'(?:\()[a-z]+(,[a-z]+)*(?:\))', str(prop.name))
                    local_entities = tuple([fol.Variable(elem) for elem in re_entities.group(0).strip('(').strip(')').split(',')])
                    assert prop.name.constants() == set(local_entities) # emergency check that parsing works
                    universe[typename][local_predicate.name].add(local_entities)
                else:
                    universe[typename][local_predicate.name].update(local_entities)

        return universe

    #TODO: SLOPPY AND NOT GENERAL, only works for <e, <e, t>
    def additionaltypes(self):
        for pred in self.universe[T_PARSE('<e, <e, t>>')]:
            first_e = set([tup[0] for tup in self.universe[T_PARSE('<e, <e, t>>')][pred]])
            second_e = set([tup[1] for tup in self.universe[T_PARSE('<e, <e, t>>')][pred]])
            for e1 in first_e:
                for e2 in second_e:
                    self.universe[T_PARSE('<e, t>')][f"{str(pred)}({str(e1)})"].add(e2)


    def readSpace(self, file):
        """Read a space from a .observations file
        """
        model_df = pd.read_csv(file, sep=' ', header=0)
        model_matrix = np.array(model_df)
        names = list(model_df.columns)
        names2 = [flip(PARSE(name)) for name in names]
        names = names2
        prop_list = [(tuple(np.array(model_df.iloc[:, index])), prop) for index, prop in enumerate(names)]
        Set = set([MeaningVec(np.array(model_df.iloc[:, index]), name=prop) for index, prop in enumerate(names)])
        return model_matrix, Set, prop_list

    def canonicalSpace(self, props):
        """ Generates a fully informative space from a set of propositions (all combinations of truth values)
        """
        space = [list(i) for i in product([0,1], repeat=len(props))]

        model_df = pd.DataFrame.from_records(space)
        names = props

        model_matrix = np.array(model_df)
        prop_list = [(tuple(np.array(model_df.iloc[:, index])), prop) for index, prop in enumerate(names)]
        Set = set([MeaningVec(np.array(model_df.iloc[:, index]), name=prop) for index, prop in enumerate(names)])

        return model_matrix, Set, prop_list

    def prettyprint(self):
        rownames = [f"M{i}" for i, row in enumerate(self.matrix)]
        columnnames = [vector.name for vector in list(self.propositions)]
        df = pd.DataFrame(self.matrix, index=rownames, columns=columnnames)
        print(df)

    def legal_entities(self, ex: fol.Expression, var: fol.Variable) -> set:
        relevant_pred, arguments = find_predicate(str(ex), var)
        local_entities = tuple([fol.Variable(elem) for elem in arguments.strip('(').strip(')').split(',')])
        variable_location = local_entities.index(var)
        argarity = len(local_entities)
        predicate_type = ex.typecheck(signature=self.type_signature)[relevant_pred]
        possible_args = list(self.universe[predicate_type][relevant_pred])
        if argarity > 1:
            legals = [tup[variable_location] for tup in possible_args]
        else:
            legals = [tup for tup in possible_args]
        return set(legals)

    def generate_dummy(self, ex: fol.Expression, var: fol.Variable) -> fol.Variable:
        parsed_var = PARSE(var.name)
        relevant_pred, arguments = find_predicate(str(ex), var)
        if str(relevant_pred).startswith('F'): #TODO: NLTK type inferencer is fucking black magic, why does it not know what type this is
            predicate_type = T_PARSE('<e, t>')
        else:
            predicate_type = ex.typecheck(signature=self.type_signature)[relevant_pred]
        if isinstance(parsed_var, fol.FunctionVariableExpression):
            possible_funcs = list(self.universe[predicate_type].items())
            dummy = possible_funcs[0][0]
        else:
            legals = list(self.legal_entities(ex, var))
            dummy = legals[0]

        return dummy

    def is_atomic(self, expression: fol.Expression) -> bool:
        """Returns true if expression is an atomic proposition
        """
        if expression in [v.name for v in self.propositions]:
            return True
        else:
            return False

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
            return MeaningVec(list(np.zeros(self.matrix.shape[0], dtype=int)), name=expression)

        #this function is memoized with the use of self.prop_dict
        if str_expression not in self.prop_dict:
            if isinstance(expression, fol.NegatedExpression):
                operator = f'{SYM.NOT}'
                term = expression.term
            else:
                try: #see if its a logical connective
                    operator = expression.getOp()
                    first_term = expression.first
                    second_term = expression.second
                except AttributeError: #otherwise maybe a quantifier
                    operator = expression.getQuantifier()
                    variable = expression.variable
                    term = expression.term
                    #introduce dummy variable, since it will be closed anyway, always choose a unique dummy consistent with interpretation function
                    dummy = self.generate_dummy(term, variable)
                    term = term.replace(variable, PARSE(str(dummy)))

        #the recursive step
            if operator == f'{SYM.NOT}':
                self.prop_dict[str_expression] = negation(self.infer_meaningvec(term))
            elif operator == f'{SYM.AND}':
                self.prop_dict[str_expression] = conjunction(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.OR}':
                self.prop_dict[str_expression] = disjunction(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.IMP}':
                self.prop_dict[str_expression] = implication(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.IFF}':
                self.prop_dict[str_expression] = equivalence(self.infer_meaningvec(first_term), self.infer_meaningvec(second_term))
            elif operator == f'{SYM.EXISTS}':
                self.prop_dict[str_expression] = self.quantify(self.infer_meaningvec(term),  str(dummy), mode='exist')
            elif operator == f'{SYM.ALL}':
                self.prop_dict[str_expression] = self.quantify(self.infer_meaningvec(term), str(dummy), mode='univ')

        return self.prop_dict[str_expression]

    def assignment_function(self, phi: MeaningVec, c: fol.Variable) -> List[MeaningVec]:
        """
        Returns the list of Meaning Vectors of proposition phi according to the assignment function g[c/x].
        Only replaces with entities in the interpretation function of phi
        List: [phi_g[c/e1], phi_g[c/e2], ... , phi_g[c/en]]
        """
        entity_names = [e.name for e in self.universe[T_PARSE('e')] if e in self.legal_entities(phi.name, c)]
        predicates = [phi.name.replace(c, PARSE(e)) for e in entity_names]
        #TODO: quick and dirty
        vecs = [self.infer_meaningvec(pred) for pred in predicates]
        return vecs #only take into account atomic non-zero vectors

    def proto_assignment_function(self, phi: MeaningVec, c: fol.Variable) -> List[MeaningVec]:
        #do you want to replace a term of type e or higher order?
        replacee = PARSE(str(c))
        typename = T_PARSE(self.type_signature[str(c)])
        if isinstance(c, fol.FunctionVariableExpression) or typename != T_PARSE('e'):
            domain = self.universe[typename]
            domain_names = [key for key, value in domain.items()]
        elif isinstance(c, fol.IndividualVariableExpression) or typename == T_PARSE('e'):
            domain = self.universe[T_PARSE('e')]
            domain_names = [e.name for e in self.universe[T_PARSE('e')] if e in self.legal_entities(phi.name, c)]

        predicates = [phi.name.replace(fol.Variable(str(c)), PARSE(e)) for e in domain_names]
        vecs = [self.infer_meaningvec(pred) for pred in predicates]
        return vecs


    def quantify(self, expression: MeaningVec, var: str, mode: str) -> MeaningVec:
        """
        Existentially binds a propositional meaning vector for some variable.
        """
        #detect to be bound constant/free variable, pick a new variable name
        to_be_bound = fol.Variable(var)
        typename = T_PARSE(self.type_signature[str(var)])

        if typename != T_PARSE('e'):
            newvar = fol.unique_variable(pattern=fol.Variable('Q'))
        else:
            newvar = fol.unique_variable(pattern=fol.Variable('x'))

        #make sure variable to be bound is valid
        assert expression.name is not None
        #assert to_be_bound in expression.name.constants().union(expression.name.predicates()), "Variable does not occur in expression"

        #look up all the vectors associated with new assignment
        vectors = self.proto_assignment_function(expression, to_be_bound)
        # Take dis/conjunction over assignment
        # If vectors is empty, that means that the assignment function returns only null-vectors and the result is also necessarily the null-vector
        if mode == 'exist':
            newname = PARSE(f'{SYM.EXISTS} {newvar}. ({expression.name})')
            if not vectors:
                return MeaningVec(list(np.zeros(self.matrix.shape[0], dtype=int)), name=newname.replace(to_be_bound, PARSE(newvar.name), alpha_convert=False))
            else:
                quantified = reduce(disjunction, vectors)
        elif mode == 'univ':
            newname = PARSE(f'{SYM.ALL} {newvar}. ({expression.name})')
            if not vectors:
                return MeaningVec(list(np.zeros(self.matrix.shape[0], dtype=int)), name=newname.replace(to_be_bound, PARSE(newvar.name), alpha_convert=False))
            else:
                quantified = reduce(conjunction, vectors)

        else:
            raise TypeError('Invalid mode, must be either \'exist\' or \'univ\'')

        quantified.name = newname.replace(to_be_bound, PARSE(newvar.name), alpha_convert=False)
        return quantified

    def infer_meaningset(self, expression: fol.Expression) -> MeaningSet:
        vectors = []
        meaningvec = self.infer_meaningvec(expression)
        for p in self.propositions:
            if entails(p, meaningvec):
                vectors.append(p)
        return set(vectors)

    def getSemantics(self, word: str) -> MeaningSet:
        """
        Get the MeaningSet for a word in the meaning space using disjunctive semantics
        """
        relevantProps = [MeaningVec(vector, name=prop) for vector, prop in self.prop_list if word in prop]
        return set(relevantProps)

    def writevectors(self, props: list, outname: str):
        with open(f"{outname}.vectors", "w") as f:
            writer = csv.writer(f, delimiter=' ')
            for prop in props:
                l = list(prop.vec)
                l.insert(0, str(prop.name).replace(',', '('))
                #stringvec = str(prop.vec).strip('[').strip(']').replace('\n', ' ')
                writer.writerow(l)
