from parsing import *
from dfs_semantics import *
from MeaningVec import *
import matplotlib.pyplot as plt
import sys

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


relevant_propositions = {'enter(john,bar)' : world.prop_dict['enter(john,bar)'],
                        'enter(john,restaurant)': world.prop_dict['enter(john,restaurant)'],
                        'enter(john,restaurant) & order(john,wine)' : world.infer_meaningvec(PARSE('(enter(john,restaurant) & order(john,wine))')),
                        'enter(john,restaurant) & order(john,beer)' : world.infer_meaningvec(PARSE('(enter(john,restaurant) & order(john,beer))'))
                        }

labels = [key for key in list(relevant_propositions.keys())]

sent = [AND, ORDER(wine), ORDER,  AND(ORDER(wine)), ENTER, ENTER(restaurant), (AND(ORDER(wine)))(ENTER(restaurant)), (AND(ORDER(wine))(ENTER(restaurant)))(john)]

a = []
for pred in sent:
    l = []
    for p in list(relevant_propositions.values()):
        try:
            l.append(inference(p, pred.simplify().real()))
        except:
            l.append(0)
    l = np.array(l).reshape(4, 1)
    a.append(l)


for i in range(len(a)):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    map = plt.get_cmap('RdYlGn')
    mat = ax.matshow(a[i].T, cmap=map, vmin=-1, vmax=1)
    ax.set_xticks(np.arange(4))
    ax.set_yticks([])
    ax.set_title(str(sent[i].simplify()))
    ax.set_xticklabels(labels, rotation=18, fontsize=6)
    ax.set_yticklabels([])
    ax.tick_params(axis = "y", which = "both", bottom = False, top = False)
    n = (str(sent[i].simplify())).replace(' ', '_')
    plt.savefig(f'plots/{n}_reversed.pdf')
