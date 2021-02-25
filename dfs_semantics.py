import csv
import numpy as np
import pandas as pd
from itertools import product

def matrix_to_set(Matrix):
    return set(map(tuple, Matrix.T))

def set_to_matrix(Set):
    array = np.array([elem for elem in Set])
    return array

def negateVector(vec):
    return vec^1

def vector_imp(A, B):
    """ Returns a vector of the elementwise implication of A --> B
    """
    return np.logical_or(negateVector(A), B)

def vector_and(A, B):
    return np.logical_and(A, B)


class MeaningSpace:
    def __init__(self, file=None, propositions=None):
        if file:
            self.matrix, self.prop_dict = self.readSpace(file)
            self.propositions = list(self.prop_dict.values())
        elif propositions:
            self.matrix, self.prop_dict = self.canonicalSpace(propositions)
            self.propositions = list(self.prop_dict.values())

    def __len__(self):
        return len(self.prop_dict)

    def __str__(self):
        return f"{set([self.prop_dict[elem] for elem in matrix_to_set(self.matrix)])}"

    def readSpace(self, file):
        model_df = pd.read_csv(file, sep=' ', header=0)
        model_matrix = np.array(model_df)
        prop_dict = list(model_df.columns)
        prop_dict = {tuple(np.array(model_df.iloc[:, index])) : prop for index, prop in enumerate(prop_dict)}
        return model_matrix, prop_dict

    def canonicalSpace(self, props):
        space = [list(i) for i in product([0,1], repeat=len(props))]

        model_df = pd.DataFrame.from_records(space)
        model_df.columns = props
        model_matrix = np.array(model_df)
        prop_dict = {tuple(np.array(model_df.iloc[:, index])) : prop for index, prop in enumerate(list(model_df.columns))}

        return model_matrix, prop_dict

    def printSet(self, Set):
        try:
            print(set([self.prop_dict[elem] for elem in Set]))
        except KeyError:
            print(set([f'v_{i}' for i, elem in enumerate(Set)]))

class Semantics:
    def __init__(self, file=None, propositions=None):
        if file:
            self.space = MeaningSpace(file)
        elif propositions:
            self.space = MeaningSpace(file=None, propositions=propositions)

    def getWordSemantics(self, word):
        """ Returns an array of all propositions (as vectors) that contain word
            word: string
        """
        relevantProps = [index for index, prop in enumerate(self.space.propositions) if word in prop]
        vectors = self.space.matrix[:, relevantProps] #TODO: Inherit numpy array slicing ?
        return vectors

    def entails(self, a, b): #only use this on vectors
        """ Returns True if the the vector implication of a --> b contains only True (ie. entailment) and False otherwise
            a: vector, b: vector
        """
        return True if np.all(vector_imp(a, b)) else False

    def setAnd(self, A, B):
        """ Returns a set containing the vector conjunction of the set A and B using the cartesian product.
            A: array, B: array
        """
        A = [column for column in A.T]
        B = [column for column in B.T]
        C = []
        pairs = [pair for pair in product(A, B)]
        for first, second in pairs:
            C.append(vector_and(first, second))

        return np.array(C).T

    def setMerge(self, A, B): #transpositions are to get columns/rows correctly
        """ Returns the set of vectors in B that entail one or more vectors in A
            A: numpy array (functor), B: numpy array (argument)
        """
        # i loop over vectors because I am not sure how this works on the matrix-level (differently sized matrices etc.)
        merged_meaning = []
        for i, argument_p in enumerate(B.T):
            for j, functor_p in enumerate(A.T):
                if self.entails(argument_p, functor_p):
                    merged_meaning.append(argument_p)
                    break
        return np.array(merged_meaning).T
