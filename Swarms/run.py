from __future__ import annotations

import pickle
import shelve
from typing import  TypeVar, Callable, Iterator
from dataclasses import dataclass

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import numpy as np
from numpy.typing import NDArray
import seaborn as sns
from scipy.interpolate import RBFInterpolator
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis, LinearDiscriminantAnalysis

sns.set_theme(
    style = 'darkgrid',
    #  notebook, paper, talk, poster
    context = 'paper',
    # deep, muted, bright, pastel, dark, colorblind
    palette = 'muted',
    color_codes = True,
    font = 'serif',
    rc = {
        'axes.labelpad': 10,
        'savefig.bbox': 'tight',
        'figure.autolayout': True,
        'figure.figsize': (3.0, 2.4),
        'font.family': ['serif'],
        'font.size': 11,
        'axes.titlesize': 'large',
        'axes.labelsize': 'medium',
        'xtick.labelsize': 'small',
        'ytick.labelsize': 'small',
        'text.usetex': True,
        'pgf.rcfonts': False,
        'pgf.preamble':  r"""
            \usepackage[brazilian]{babel}
            \usepackage[T1]{fontenc}
            \usepackage[utf8]{inputenc}
        """,
    },
)

T = TypeVar('T')

def parse(text: str, *, type: Callable[[str], T] = str) -> list[T]:
    return [type(s.strip()) for s in text.strip().strip('[]').split(',')]

def content_lines(path: str, /) -> Iterator[str]:
    with open(path, 'rt') as file:
        for line in file:
            line = line.strip()
            if len(line) > 0:
                yield line

def show(*args) -> None:
    # print(*args)
    pass

def summary(x: NDArray[np.float64]) -> str:
    x = np.asarray(x)
    if len(x) == 0:
        return '<empty>'
    return f'min = {np.min(x)} / mean = {np.mean(x)} / max = {np.max(x)} / std = {np.std(x)} / shape = {np.shape(x)}'


db = shelve.open('cache.db', flag='c', protocol=pickle.HIGHEST_PROTOCOL, writeback=False)

@dataclass(frozen=True)
class Edge:
    s: int
    p: int
    k: int

    @property
    def latex(self) -> str:
        return f'$e_{{{self.s},{self.p},{self.k}}}$'

    def __str__(self) -> str:
        return f'e_{self.s},{self.p},{self.k}'

    @staticmethod
    def parse(text: str):
        e, s, p, k = text.strip().split('_')
        assert e == 'e', f'wrong edge: {text}'
        return Edge(int(s), int(p), int(k))


class EdgeSet(dict[str, int]):
    def __init__(self, edges: tuple[Edge]):
        super().__init__()
        self.edges = edges

    def load(self, filename: str = 'solutions.txt') -> None:
        for line in content_lines(filename):
            key, blocks = line.split(',')
            key = key.strip()
            blocks = int(blocks)
            assert key not in self, f"repeated solution: {key} ({blocks})"
            self[key] = blocks

    @staticmethod
    def init(filename: str = 'edges.txt') -> EdgeSet:
        try:
            return db['EdgeSet:init']
        except:
            print('Could not load edge set')

        edges = [Edge.parse(line) for line in content_lines(filename)]
        es = EdgeSet(edges)
        es.load()

        db['EdgeSet:init'] = es
        return es

edgeSet = EdgeSet.init()
print('blocks(EdgeSet):', summary([w for w in edgeSet.values()]))

@dataclass(frozen=True)
class Particle:
    weight: NDArray[np.float64]
    edges: list[Edge]
    left: list[str]
    right: list[str]
    velocity: NDArray[np.float64]
    selected: NDArray[np.bool_]

    @property
    def selected_edges(self) -> list[Edge | None]:
        return [edge if selected else None for edge, selected in zip(edgeSet.edges, self.selected, strict=True)]

    @staticmethod
    def load(filename: str = 'iters.txt') -> list[list[Particle]]:
        try:
            return db['Particle:load']
        except:
            print('Could not load particles')

        lines = content_lines(filename)
        iteration = 0
        swarm = list[Particle]()

        result = list[list[Particle]]()
        for line in lines:
            line = line.strip()
            if len(line) <= 0:
                continue

            if line.startswith('Iteration:'):
                _, inum = line.split()
                inum = int(inum)
                if inum > 0:
                    iteration += 1

                assert iteration == int(inum), f"wrong iteration: {inum} != {iteration}"
                if iteration > 0:
                    result.append(swarm)
                    swarm = []
                continue

            weight = np.asarray(parse(line, type=float))
            velocity = np.asarray(parse(next(lines), type=float))
            edges = parse(next(lines), type=Edge.parse)
            left = parse(next(lines))
            right = parse(next(lines))
            selected = np.asarray(parse(next(lines), type=lambda s: s == 'True'))

            swarm.append(Particle(weight, edges, left, right, velocity, selected))

        result.append(swarm)
        db['Particle:load'] = result
        return result

swarms = Particle.load()
total_weights = np.stack([p.weight for swarm in swarms for p in swarm])
print('total_weight:', summary(total_weights))

def gen_sample(*, n: int = 10, w: float = np.max(np.abs(total_weights))) -> tuple[NDArray[np.float64], NDArray[np.uint8]]:
    try:
        return db['sample:points'], db['sample:blocks']
    except:
        print('Could not load sample')

    dtype = np.float64
    base = np.linspace(-w, w, num=n, endpoint=True, dtype=dtype)
    points = np.zeros((), dtype=dtype)

    for _ in range(len(edgeSet.edges)):
        try:
            p = len(points)
        except TypeError:
            points = base[..., np.newaxis]
            continue

        rows = list[NDArray[dtype]]()
        for val in base:
            new = np.full((p, 1), val, dtype=dtype)
            row = np.concatenate((new, points), axis=1, dtype=dtype)
            rows.append(row)
        points = np.concatenate(rows, dtype=dtype)

    rng = np.random.default_rng()
    points = points + rng.normal(0, 1e-4, size=points.shape).astype(dtype)

    blocks = list[int]()
    for row in points:
        key = ''.join(str(i) for i in np.argsort(row))
        blocks.append(edgeSet[key])
    blocks = np.asarray(blocks, dtype=np.uint8)

    db['sample:points'] = points
    db['sample:blocks'] = blocks
    return points, blocks

sample, blocks = gen_sample()
print('sample(7D):', summary(sample))
print('blocks(sample):', summary(blocks))

def fit_pca() -> PCA:
    try:
        pca = db['pca:fit']
    except:
        print('Could not load PCA')
        pca = PCA(n_components=2, svd_solver='full')
        pca.fit(sample)
        db['pca:fit'] = pca

    print('pca-classes:', *pca.classes_)
    return pca

def fit_qda() -> QuadraticDiscriminantAnalysis:
    try:
        qda = db['qda:fit']
    except:
        print('Could not load QDA')
        qda = QuadraticDiscriminantAnalysis()
        qda.fit(sample, blocks)
        db['qda:fit'] = qda

    print('qda-classes:', *qda.classes_)
    return qda

def fit_lda() -> QuadraticDiscriminantAnalysis:
    try:
        lda = db['lda:fit']
    except:
        print('Could not load LDA')
        lda = LinearDiscriminantAnalysis(n_components=2, solver='eigen')
        lda.fit(sample, blocks)
        db['lda:fit'] = lda

    print('lda-classes:', *lda.classes_)
    return lda

lda = fit_lda()
transformed = lda.transform(sample).astype(np.float64)
print('sample(LDAd):', summary(transformed))

for b in range(np.max(blocks) + 1):
    idx = blocks == b
    bn = np.count_nonzero(idx)
    if bn > 0:
        print(f'blocks == {b}:', bn)
        print('x:', summary(transformed[idx,0]))
        print('y:', summary(transformed[idx,1]))

def block_avg(x: NDArray[np.float64]) -> NDArray[np.float64]:
    d = x.ndim - 1
    assert x.shape[d] == len(lda.classes_), f"shape {x.shape} does not match blocks"

    for bi, b in enumerate(lda.classes_):
        x[..., bi] *= b
    x = np.sum(x, axis=d)
    return x

def fit_interp() -> Callable[[NDArray[np.float64], NDArray[np.float64]], NDArray[np.float64]]:
    # qda = fit_qda()

    prob = lda.predict_proba(sample).astype(np.float64)
    avg_blocks = block_avg(prob).astype(np.float64)
    print('blocks(LDA):', summary(avg_blocks))
    try:
        interp = db['block:interp']
    except:
        print('Could not load block interpolation')
        interp = RBFInterpolator(transformed, avg_blocks, neighbors=200, kernel='cubic', smoothing=1)
        db[f'block:interp'] = interp

    def ev(x: NDArray[np.float64], y: NDArray[np.float64]) -> NDArray[np.float64]:
        return interp(np.c_[x.ravel(), y.ravel()]).reshape(x.shape)

    return ev

def interp_mesh() -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    interp = fit_interp()
    try:
        X = db['interp_mesh:X']
        Y = db['interp_mesh:Y']
        Z = db['interp_mesh:Z']
    except:
        print('Could not load mesh X, Y, Z')
        wx, wy = lda.transform(total_weights).T

        xmin, xmax = np.min(wx), np.max(wx)
        ymin, ymax =  np.min(wy), np.max(wy)
        dx = (xmax - xmin) / 10
        dy = (ymax - ymin) / 10
        xmin, xmax = xmin - dx, xmax + dx
        ymin, ymax = ymin - dy, ymax + dy

        X = np.linspace(xmin, xmax, num=320, endpoint=True)
        Y = np.linspace(ymin, ymax, num=320, endpoint=True)
        X, Y = np.meshgrid(X, Y)
        db['interp_mesh:X'] = X
        db['interp_mesh:Y'] = Y
        Z = None

    print(f'X(mesh):', summary(X))
    print(f'Y(mesh):', summary(Y))

    if Z is None:
        Z = interp(X, Y)
        db['interp_mesh:Z'] = Z

    print(f'Z(mesh):', summary(Z))
    return X, Y, Z


X, Y, Z = interp_mesh()
levels = np.linspace(np.min(lda.classes_), np.max(lda.classes_), num=3+2*50, endpoint=True)

cmap = mpl.colormaps[mpl.rcParams["image.cmap"]].reversed()

# for i in range(len(swarms) - 1):
#     print()
#     print('swarm', i)

#     plt.clf()
#     w = lda.transform([p.weight for p in swarms[i]]).T
#     v = lda.transform([p.velocity for p in swarms[i + 1]]).T
#     print(f'w({i}):', summary(w))
#     print(f'v({i}+1):', summary(v))

#     plt.quiver(w[0], w[1], v[0], v[1], color=dotc, alpha=0.7, clip_on=True, zorder=1.5)
#     plt.scatter(w[0], w[1], color=dotc, zorder=1.6)

#     for pi, wx, wy in zip(swarms[i], w[0], w[1], strict=True):
#         edges = '\n'.join(e.latex for e in pi.selected_edges if e is not None)
#         plt.annotate(edges, (wx, wy), fontsize='x-small', color=dotc, clip_on=True, zorder=1.7)

#     csf = plt.contourf(X, Y, Z, levels=levels, cmap=cmap, zorder=1)
#     cbar = plt.colorbar(csf, label='Média de Blocos')
#     # cbar.add_lines(levels=lda.classes_, colors=['white']*len(lda.classes_), linewidths=1.0)

#     # for b, c in zip(lda.classes_, ('b', 'r', 'k'), strict=True):
#     #     plt.scatter(transformed[blocks == b,0], transformed[blocks == b,1], color=c, alpha=0.01, label=f'Blocos = {b}')
#     # plt.legend(loc='lower right')
#     # plt.xlim(x_min, x_max)
#     # plt.ylim(y_min, y_max)

#     plt.tick_params(axis='both', which='both',
#         bottom=False, top=False, labelbottom=False, labeltop=False,
#         left=False, right=False, labelleft=False, labelright=False)
#     plt.title(f'Iteração {i}')

#     plt.savefig(f'out/plot-{i:02d}.png')
#     plt.savefig(f'out/iter-{i:02d}.pgf')


plt.clf()
fig = plt.figure()
fig, ax = plt.subplots(1, 3,
    sharex=True, sharey=True,
    squeeze=True,
    layout='tight', figsize=(12, 3.6), width_ratios=[1, 1, 1.2])

for axi, i in zip(ax, [1, 5, 9], strict=True):
    assert isinstance(axi, Axes)
    w = lda.transform([p.weight for p in swarms[i]]).T
    v = lda.transform([p.velocity for p in swarms[i + 1]]).T

    axi.quiver(w[0], w[1], v[0], v[1], color='black', alpha=0.7, clip_on=True, zorder=1.5)
    axi.scatter(w[0], w[1], color='black', zorder=1.6)

    for pi, wx, wy in zip(swarms[i], w[0], w[1], strict=True):
        edges = '\n'.join(e.latex for e in pi.selected_edges if e is not None)
        axi.annotate(edges, (wx, wy), fontsize='x-small', color='white', clip_on=True, zorder=1.7)

    axi.set_title(f'Iteração {i}')
    csf = axi.contourf(X, Y, Z, levels=levels, cmap=cmap, zorder=1)

    axi.tick_params(axis='both', which='both',
        bottom=False, top=False, labelbottom=False, labeltop=False,
        left=False, right=False, labelleft=False, labelright=False)

plt.colorbar(csf, label='Média de Blocos', ax=ax[-1])
plt.savefig(f'out/plot-multi.png')
plt.savefig(f'out/iter-multi.pgf')
