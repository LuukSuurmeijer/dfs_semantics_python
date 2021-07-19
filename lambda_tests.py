from dfs_semantics import *
import pandas as pd
from itertools import product
from nltk import Tree


# World
world = MeaningSpace(file='wollic.observations')

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
AND = BinaryOperator(PARSE('\\P \\Q \\x. (P(x) & Q(x))'), world, setconjoin)
OR = BinaryOperator(PARSE('\\P \\Q \\x. (P(x) | Q(x))'), world, setdisjoin1)
OR_functional = BinaryOperator(PARSE('\\P \\Q \\x. (P(x) | Q(x))'), world, lambda a, b: setnegate(setconjoin(setnegate(a), setnegate(b))))

first = (OR(LEAVE)(ASKMENU))(ellen).real()
second = (OR_functional(LEAVE)(ASKMENU))(ellen).real()
# is functional completeness the same as disjunction?
assert (OR(LEAVE)(ASKMENU))(ellen).close() == (OR_functional(LEAVE)(ASKMENU))(ellen).close()
# are both the same as direct propositional disjunction?
assert first == second == world.infer_meaningvec(PARSE('leave(ellen) | askmenu(ellen)'))
print(':-)')



and2 = AND(LEAVE).simplify()
print(and2, ' | ', and2.close())
and3 = (AND(LEAVE)(ASKMENU).simplify())
print(and3, ' | ', and3.close())
and4 = (AND(LEAVE)(ASKMENU))(ellen).simplify()
print(and4, ' | ', and4.close())

print('=======================================================================')

not2 = NOT(LEAVE).simplify()
not3 = NOT(LEAVE)(ellen).simplify()
print(not2.close())
print('\n')
print(not3.close())

input = f"""[{str(and4).replace(' ', '')}
            [ellen] [{str(and3).replace(' ', '')}
            [{str(ASKMENU).replace(' ', '')}]
            [{str(and2).replace(' ', '')} [{str(AND).replace(' ', '')}] [{str(LEAVE).replace(' ', '')}]]]]"""
tree = Tree.fromstring(input, brackets='[]')
