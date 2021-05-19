from dfs_semantics import *
from type import *
from sklearn.manifold import MDS
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import sys

world = MeaningSpace(file='model.observations')

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions, lambda x: x)

def set_average(set):
    matrix = set_to_matrix(set)
    return np.sum(matrix, axis=1, keepdims=True) / len(set)


#Lexicon
mike = getSemantics(world, 'mike')
jess = getSemantics(world, 'jess')
read = lambda x: merge(getSemantics(world, 'read'), x)
tease = lambda x: lambda y: merge(merge(getSemantics(world, 'tease'), x), y)


full = set_to_matrix(list(tease(jess)(mike))[1]).reshape(-1,1) #TODO: DIRECTIONALITY !!!! This set contains both tease(jess, mike) and tease(mike, jess)


tease = set_average(getSemantics(world, 'tease'))
world.propositions.append('JESS')
tease_jess = set_average(merge(getSemantics(world, 'tease'), jess))
world.propositions.append('TEASE(JESS)')
tease_jess_mike = full
world.propositions.append('')

world_matrix = world.matrix
print(world_matrix.shape)
world_matrix = np.append(world_matrix, set_average(jess), axis=1)
world_matrix = np.append(world_matrix, tease_jess, axis=1)
world_matrix = np.append(world_matrix, tease_jess_mike, axis=1)


embedding = MDS(n_components=3)
world_3d = embedding.fit_transform(world_matrix.T).T
print(world_3d.shape)
test = world_3d[:,-4:-1]

fig = plt.figure()
ax = Axes3D(fig)
world_3d = world_3d.T
x, y, z = world_3d[:,0], world_3d[:,1], world_3d[:,2]
xvalues = []
yvalues = []
zvalues = []

ax.scatter(x, y, z)
for i, txt in enumerate(world.propositions):
    if txt.upper() == txt:
        ax.text(x[i], y[i], z[i], '%s' % (str(txt)), size=5, zorder=0, color='red')
        xvalues.append(x[i])
        yvalues.append(y[i])
        zvalues.append(z[i])
    else:
        ax.text(x[i], y[i], z[i], '%s' % (str(txt)), size=5, zorder=0)
ax.plot(xvalues, yvalues, zvalues, color='green', zorder=0, linewidth=0.7)

plt.savefig('test.pdf')
