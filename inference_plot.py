from dfs_semantics import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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

props = list(world.propositions)
for pred in predicates:
    closure = world.infer_meaningvec(pred.closure)
    props.append(closure)
labels = [str(p).replace('exists', '∃') for p in props]

def inference_matrix_plot(meaningvectors, labels):
    m = np.zeros((len(props), len(props)))
    for i in range(len(m)):
        for j in range(len(m)):
            m[i][j] = inference(props[i], props[j])

    fig = plt.figure()
    ax = fig.add_subplot(111)
    mat = ax.matshow(m, cmap=plt.get_cmap('RdYlGn'))
    fig.colorbar(mat)

    ax.set_xticks(np.arange(len(m)))
    ax.set_yticks(np.arange(len(m)))
    ax.set_xticklabels(labels, rotation=45, fontsize=4.7, ha='right')
    ax.set_yticklabels(labels, rotation=0, fontsize=4.7)
    ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
    fig.subplots_adjust(bottom=0.2)
    fig.align_xlabels()
    plt.savefig('inference_matrix_plot.pdf')
    plt.show()

def plot_probabilities(meaningvecs, labels):
    for p in predicates:
        meaningvecs.append(p.real())
        labels.append(str(p))
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.bar(labels, [vec.prob for vec in meaningvecs], color='green')
    ax.set_xticks(np.arange(len(meaningvecs)))
    ax.set_xticklabels(labels, rotation=-85, fontsize=4)
    ax.set_ylabel("Probability")
    plt.show()

inference_matrix_plot(props, labels)
card_ex = []
prob_ex = []
prob_avg = []
dists = []
names = []
for pred in predicates:
    names.append(str(list(pred.denotation.predicates())[0]))
    s = pred.close()
    cardinality = len(s)
    closure = world.infer_meaningvec(pred.closure)
    avg = pred.real()
    card_ex.append(cardinality)
    prob_ex.append(closure.prob)
    prob_avg.append(avg.prob)
    dists.append(float(np.dot(closure, avg)/(np.linalg.norm(closure)*np.linalg.norm(avg))))

card_dis = []
prob_dis = []
for pred in predicates:
    name = str(list(pred.denotation.predicates())[0])
    names.append(name)
    s = world.getSemantics(name)
    cardinality = len(s)
    avg = set_average(s)
    card_dis.append(cardinality)
    prob_dis.append(avg.prob)

#df = pd.DataFrame(list(zip(names, card_ex, card_dis, prob_ex, prob_dis, prob_avg, dists)), columns =['name', 'card_ex', 'card_dis', 'prob_closure', 'prob_dis_avg', 'prob_ex_avg', 'distance'])
#print(df)

inference_matrix_plot(props, labels)
