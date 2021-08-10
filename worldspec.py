from dfs_semantics import *
from parsing import *
from MeaningVec import *
import pandas as pd
from itertools import product
from nltk import Tree


# World
#TODO: for now you have to define the types of variables you will be using because I dont know how NLTK's fucking type inferencer works
typesig = {'leave' : '<e, t>', 'askmenu' : '<e, t>', 'pay' : '<e, t>', 'P' : '<e, t>', 'Q' : '<e, t>', 'R' : '<e, <e, t>>', 'S' : '<e, <e, t>>',
        'order' : '<e, <e, t>>', 'drink' : '<e, <e, t>>', 'enter' : '<e, <e, t>>', 'eat' : '<e, <e, t>>'}
world = MeaningSpace(file='worlds/wollic.observations', signature=typesig)
world.additionaltypes()
#operators
AND = MeaningSet(PARSE('\\P \\Q \\x. (P(x) & Q(x))'), world)

#predicates
PAY = MeaningSet(PARSE('\\x. (pay(x))'), world)
LEAVE = MeaningSet(PARSE('\\x. (leave(x))'), world)
DRINK = MeaningSet(PARSE('\\y \\x. (drink(y, x))'), world)
EAT = MeaningSet(PARSE('\\y \\x. (eat(y, x))'), world)
ENTER = MeaningSet(PARSE('\\y \\x. (enter(y, x))'), world)
ORDER = MeaningSet(PARSE('\\y \\x. (order(y, x))'), world)
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

def setconjoin(a: Set[MeaningVec], b: Set[MeaningVec]) -> Set:
    cart = [conjunction(first, second) for first, second in product(a, b)]
    return set(cart)

def setnegate(a: Set[MeaningVec]) -> Set:
    return set([negation(prop) for prop in a])

def setdisjoin1(a: Set[MeaningVec], b: Set[MeaningVec]) -> Set:
    cart = [disjunction(first, second) for first, second in product(a, b)]
    return set(cart)

def setdisjoin2(a: Set[MeaningVec], b: Set[MeaningVec]) -> Set:
    return setnegate(setconjoin(setnegate(a), setnegate(b)))


NOT = UnaryOperator(PARSE('\\P \\x. (-(P(x)))'), world, lambda X: set([negation(prop) for prop in X]))
ETET = UnaryOperator(PARSE('\\P \\x. (P(x))'), world, lambda X: set([prop for prop in X]))
AND = BinaryOperator(PARSE('\\P \\Q \\x. (P(x) & Q(x))'), world, setconjoin)
OR = BinaryOperator(PARSE('\\P \\Q \\x. (P(x) | Q(x))'), world, setdisjoin1)
OR_functional = BinaryOperator(PARSE('\\P \\Q \\x. (P(x) | Q(x))'), world, lambda a, b: setnegate(setconjoin(setnegate(a), setnegate(b))))
EVERYONE = Quantifier(PARSE('\\P. all x. P(x)'), world, setconjoin)
