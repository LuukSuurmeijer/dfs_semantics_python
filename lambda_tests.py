from dfs_semantics import *
import pandas as pd
from itertools import product

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

def setconjoin(a: MeaningSet, b: MeaningSet, world: MeaningSpace) -> MeaningSet:
    cart = [conjunction(first, second) for first, second in product(a.close(), b.close())]
    denotation = PARSE(str(a.denotation) + SYM.AND + str(b.denotation))
    return MeaningSet(denotation, world, cart)

def setnegate(a: MeaningSet, world: MeaningSpace) -> MeaningSet:
    denotation = PARSE(f"{SYM.NOT}({str(a.denotation)})")
    return MeaningSet(denotation, world, set([negation(prop) for prop in a.close()]))

print(world.infer_meaningvec(PARSE('(-(leave(john)))')))

NOT = UnaryOperator(PARSE('\\P \\x. (-(P(x)))'), world, lambda X: set([negation(prop) for prop in X]))
not2 = NOT(LEAVE).simplify()
not3 = NOT(LEAVE)(ellen).simplify()

print(type(NOT))
print(NOT.close())
print(type(not2))
print(not2.close())
print(type(not3))
print(not3.close())
