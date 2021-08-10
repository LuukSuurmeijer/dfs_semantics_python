from parsing import *
from dfs_semantics import *
from MeaningVec import *
import matplotlib.pyplot as plt
import sys

from worldspec import *

def treeplot(relevant_props, logical_form):
    a = []
    for pred in logical_form:
        l = []
        for p in list(relevant_props.values()):
            l.append(inference(p, MeaningVec(pred.simplify().real(), name=None)))
            #l.append(inference_raw(p.vec, pred.simplify().real()))
        l = np.array(l).reshape(len(relevant_props.items()), 1)
        a.append(l)
    print(a)
    for i in range(len(a)):
        fig = plt.figure()
        ax = fig.add_subplot(111)
        map = plt.get_cmap('RdYlGn')
        mat = ax.matshow(a[i].T, cmap=map, vmin=-1, vmax=1)
        ax.set_xticks(np.arange(len(relevant_propositions.items())))
        ax.set_yticks([])
        ax.set_title(str(logical_form[i].simplify()).replace('\\', 'λ').replace('&', '∧').replace('all', '∀').replace('exists', '∃'), fontsize=16)
        ax.set_xticklabels(np.arange(1, len(relevant_propositions.items())+1), rotation=0, fontsize=10)
        ax.set_yticklabels([])
        ax.tick_params(axis = "y", which = "both", bottom = False, top = False)
        ax.tick_params(axis="x", bottom=True, top=False, labelbottom=True, labeltop=False)
        n = (str(logical_form[i].simplify())).replace(' ', '_')
        plt.savefig(f'plots/{n}_reversed.pdf', bbox_inches='tight')



relevant_propositions = {'enter(bar,john)' : world.prop_dict['enter(bar,john)'],
                        'enter(restaurant,john)': world.prop_dict['enter(restaurant,john)'],
                        'enter(restaurant,john) & order(wine,john)' : world.infer_meaningvec(PARSE('(enter(restaurant,john) & order(wine,john))')),
                        'enter(restaurant,john) & order(beer,john)' : world.infer_meaningvec(PARSE('(enter(restaurant,john) & order(beer,john))')),
                        'drink(wine,john)' : world.infer_meaningvec(PARSE('drink(wine,john)')),
                        'eat(pizza,ellen)': world.prop_dict['eat(pizza,ellen)']
                        }

relevant_propositions_quantification = {'-eat(fries, john)' : world.infer_meaningvec(PARSE('-eat(fries,john)')),
                                        'eat(fries,ellen)' : world.prop_dict['eat(fries,ellen)'],
                                        'enter(restaurant,john)' : world.prop_dict['enter(restaurant,john)'],
                                        'enter(bar,john)' : world.prop_dict['enter(bar,john)'],
                                        'leave(john) & -pay(john)' : world.infer_meaningvec(PARSE('leave(john) & -pay(john)')),
                                        'eat(pizza,ellen) & eat(pizza,john)' : world.infer_meaningvec(PARSE('eat(pizza,ellen) & eat(pizza,john)'))
                                        }

labels = [key for key in list(relevant_propositions.keys())]
labels_quant = [key for key in list(relevant_propositions_quantification.keys())]


sent = [AND, ORDER(wine), ORDER,  AND(ORDER(wine)), ENTER, ENTER(restaurant), (AND(ORDER(wine)))(ENTER(restaurant)), (AND(ORDER(wine))(ENTER(restaurant)))(john)]
sent_quantification = [EVERYONE, ORDER, ORDER(fries), EVERYONE(ORDER(fries))]
treeplot(relevant_propositions_quantification, sent_quantification)


a = world.infer_meaningvec(PARSE('all x. order(fries,x)'))
b = world.infer_meaningvec(PARSE('(exists x. order(beer,x))'))
print(inference(b, a))


print(world.modelEntailment(a))
for prop in setnegate(world.propositions):
    if entails(a, prop):
        print(prop)
