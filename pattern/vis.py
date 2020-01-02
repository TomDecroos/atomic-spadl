from scipy import linalg
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotsoccer as mps
import numpy as np
import math

def locmovaxes(figsize=4):
    fig,axs = plt.subplots(1,2)
    fig.set_size_inches((figsize*3,figsize))
    mps.field(ax=axs[0],show=False)
    plt.axis("on")
    plt.axis("scaled")
    axs[1].axvline(0,color="black")
    axs[1].axhline(0,color="black")
    axs[1].grid()
    axs[0].set_xlim(-1,105+1)
    axs[0].set_ylim(-1,68+1)
    axs[1].set_xlim(-105,105)
    axs[1].set_ylim(-68,68)
    return axs


def _ellips(mean,covar,color=None):
    v, w = linalg.eigh(covar)
    v = 2. * np.sqrt(2.) * np.sqrt(v)
    u = w[0] / linalg.norm(w[0])
    
    # Plot an ellipse to show the Gaussian component
    angle = np.arctan(u[1] / u[0])
    angle = 180. * angle / np.pi  # convert to degrees
    ell = mpl.patches.Ellipse(mean, v[0], v[1], 180. + angle, color=color)
    #ell.set_clip_box(axs[0].bbox)
    ell.set_alpha(0.7)
    return ell


colors = ["#377eb8",
    "#e41a1c",
    "#4daf4a",
    "#984ea3",
    "#ff7f00",
    "#ffff33",
    "#a65628",
    "#f781bf",
    "#999999",
    ]

def component_ellipses(gmm,i,color=None):
    ells = []
    for a,b in [(0,2),(2,4)]:
        m,var = gmm.means_[i,a:b], gmm.covariances_[i,a:b,a:b]
        ell = _ellips(m,var,color)
        ell.width = max(ell.width,3)
        ell.height = max(ell.height,3)
        ells.append(ell)
    return ells[0],ells[1]

def mirror_ellips(ellips):
    ellips.width = 68-width
    ellips.h

def _loc_ellipses(gmm,colors=colors):
    means = [gmm.means_[i,0:2] for i in range(gmm.n_components)]
    covars = [gmm.covariances_[i,0:2,0:2] for i in range(gmm.n_components)]
    ells = [_ellips(m,c,colors[i % len(colors)]) for i,(m,c) in enumerate(zip(means,covars))]
    for ell in ells:
        ell.width = max(ell.width,3)
        ell.height = max(ell.height,3)
    return ells

def _mov_ellipses(gmm,colors=colors):
    if len(gmm.means_[0]) == 4:
        means = [gmm.means_[i,2:4] for i in range(gmm.n_components)]
        covars = [gmm.covariances_[i,2:4,2:4] for i in range(gmm.n_components)]
        return [_ellips(m,c,colors[i % len(colors)]) for i,(m,c) in enumerate(zip(means,covars))]
    else:
        return []

def plot_gmm(gmm,figsize=4,colors=colors,show=True):
    axs = locmovaxes(figsize)
    for ell in _loc_ellipses(gmm,colors):
        axs[0].add_artist(ell)
    for ell in _mov_ellipses(gmm,colors):
        axs[1].add_artist(ell)
    if show:
        plt.show()

def sample(probs):
    n = len(probs)
    return np.random.choice(n,p=probs)

def plot_gmm_actions(gmm,actions,cols,samplefn="max",figsize=4,colors=colors,show=True):
    a = actions
    labelprobs = gmm.predict_proba(a[cols])
    if samplefn == "max":
        labels = np.argmax(labelprobs,axis=1)
    else:
        labels = np.apply_along_axis(sample,axis=1, arr=labelprobs)
    pcolors = [colors[l % len(colors)] for l in labels]
    
    axs = locmovaxes(figsize)
    axs[0].scatter(a.x,a.y,color=pcolors)
    axs[1].scatter(a.dx,a.dy,color=pcolors)
    if show:
        plt.show()


def plot_components(gmm,actions,cols,samplefn="max",figsize=4,colors=colors,show=True):
    a = actions
    labelprobs = gmm.predict_proba(a[cols])
    if samplefn == "max":
        labels = np.argmax(labelprobs,axis=1)
    else:
        labels = np.apply_along_axis(sample,axis=1, arr=labelprobs)
    loc_ellipses = _loc_ellipses(gmm,colors)
    mov_ellipses = _mov_ellipses(gmm,colors)

    for i in range(gmm.n_components):
        label_idx = labels == i
        a = actions[label_idx]
        axs = locmovaxes(figsize)
        axs[0].scatter(a.x,a.y,color=colors[i % len(colors)],alpha=0.3)
        axs[1].scatter(a.dx,a.dy,color=colors[i % len(colors)],alpha=0.3)
        
        axs[0].add_artist(loc_ellipses[i])
        axs[1].add_artist(mov_ellipses[i])
        plt.show()


def plot_component_w_arrow(gmm,i,ax=None,color=None,arrowsize=2.5,mirror=False,show=True):
    if ax is None:
        ax = mps.field(show=False)

    locell,_movell = component_ellipses(gmm,i,color=color)
    if mirror:
        c = locell._center
        locell._center = [105-c[0],68-c[1]]

    ax.add_artist(locell)
    

    x,y,dx,dy = gmm.means_[i][:4]

    if mirror:
        x,y,dx,dy = 105-x,68 -y,-dx,-dy

    if abs(dx) > 0.5 or abs(dy) > 0.5:
        ax.arrow(
            x,
            y,
            dx,
            dy,
            head_width=arrowsize,
            head_length=arrowsize,
            linewidth=5,
            fc="black",#colors[i % len(colors)],
            ec="black",#colors[i % len(colors)],
            length_includes_head=True,
            alpha=1
            )
    if show:
        plt.show()
    return ax

def plot_components_arrows(gmm,figsize=6,cols=4,colors=colors,show=True):
    loc_ellipses = _loc_ellipses(gmm,colors)

    cols = cols
    rows = ((len(loc_ellipses)-1) // cols) + 1

    fig,axs = plt.subplots(rows,cols)

    if len(loc_ellipses) <= cols:
        axs = [axs]

    [ax.axis("off") for axrow in axs for ax in axrow]

    fig.set_size_inches((figsize*cols*1.5,rows*figsize))
    
    sortedcomps = np.argsort([m[0] for m in gmm.means_])

    for i,j in enumerate(sortedcomps):

        r = i // cols
        c = i % cols

        ax = axs[r][c]

        mps.field(ax=ax,show=False)
        plot_component_w_arrow(gmm,j,ax=ax,color=colors[j % len(colors)],show=False)
        ax.axis("off")

    plt.show()