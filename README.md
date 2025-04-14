# NeuralPDE for Sea Ice

Research code for solving the forced advection-diffusion inverse problem for polar sea ice concentration.


#### Note on This Document
While it is my hope that this summary of the methods and current work can always be complete, correct, and concise, it is the case that this page serves as a place to record ideas and thoughts on how to proceed, which, by their nature, can virtually never satisfies those desirable qualities.

If you spot any egregious errors, please open a bug with the details or your questions or a pull request with the correction.


#### Note on Notation
I've attempted to maintain consistent notation throughout.  However, in various places certain notations are more or less useful, and I possess not the foresight to use such notation throughout or it was unnecessarily cumbersome to use everywhere.  In these places, I have used the best notation to lead to the least ambiguity about the topics discussed.


## The Problem
We're trying to understand, quantify, and classify ice flows in Earth's polar regions.  NOAA and NASA publish large sea ice concentration datasets dating back nearly 40 years.

From observation, we believe the relevant physics to be governed by a forced advection-diffusion equation, like
```math
    u_t = \nabla \cdot \left( \kappa \nabla u \right) + \vec{v} \cdot \nabla u + f
```
where space $x \in \mathbb{R}^2$, time $t \in \mathbb{R}^+$, the diffusion field $\kappa : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$, the velocity field $(v, w)^T =: \vec{v} : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}^2$, the forcing term $f : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$ corresponding to ice freeze and thaw, and sea ice concentration $u : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow [0, 1]$.

In more words/to play with the terminology, the author posits that one could describe this problem as an "anisotropic forced advection-diffusion" problem.  Another fun way of describing this problem may be as an "inhomogeneous heterogeneous advection-diffusion" problem, (where "inhomogeneous" refers to the inhomogeneity, that is, the forcing, and "heterogeneous" refers to the spatiotemporally varying diffusion and advection.)

Note that $u$ is known (at least known on some grid in space-time.)  Accordingly, this is an inverse problem to solve for $\kappa$, $v$, and $f$.

In general, we do not know how to solve this problem.<sup>[why not?]</sup>  In this context, it is unclear if this problem is well-posed (does there exist a unique solution for each set of inputs, at least subject to reasonable assumptions?) or even well-defined (is it possible to solve this problem given the input data?)

But, like, it's probably fine.


## Notation
This is a particularly notation-heavy discussion, so let's establish some quantities.  This is going to seem exhaustive (and/or exhausting,) but the point here is to establish a pattern so we don't have to worry (too much) about the quantities considered.

I define two collections of parameters to aide in the following discussion of the notation and meaning of these paramters,
```math
\begin{aligned}
    \lambda &= ( \kappa, v, w, f )                                  & \sim & & & \text{collection of true (underlying) parameters} \\
    \hat{\lambda} &= ( \hat{\kappa}, \hat{v}, \hat{w}, \hat{f} )    & \sim & & & \text{collection of estimated parameters.}
\end{aligned}
```
where
```math
\begin{aligned}
    \kappa, \hat{\kappa}                                            & & \sim & & & \text{diffusivity (resp. true and estimated)} \\
    v, \hat{v}                                                      & & \sim & & & \text{first component of velocity (resp. true and estimated)} \\
    w, \hat{w}                                                      & & \sim & & & \text{second component of velocity (resp. true and estimated)} \\
    f, \hat{f}                                                      & & \sim & & & \text{forcing (resp. true and estimated.)} \\
\end{aligned}
```

Then, I will interpret each of the following parameters according to,
```math
\begin{aligned}
    t^n                                                             & & \sim & & & \text{the $n^{th}$ timestep, $n \in [1, \dots, N]$} \\
    x_i                                                             & & \sim & & & \text{the $i^{th}$ $x$ coordinate (position), $i \in [1, \dots, I]$} \\
    y_j                                                             & & \sim & & & \text{the $j^{th}$ $y$ coordinate (position), $j \in [1, \dots, J]$} \\
    u^n_{ij}                                                        & & \sim & & & \text{$u(t^n, x_i, y_j)$, the true (experimental) value of $u$ at $(t^n, x_i, y_j)$} \\
    \pi^n_{ij} \text{ such that } \pi \in \lambda                   & & \sim & & & \text{$\pi(t^n, x_i, y_j)$, the true (underlying) value of $\pi$ at $(t^n, x_i, y_j)$} \\
    \hat{u}^n_{ij}                                                  & & \sim & & & \text{$\hat{u}(t^n, x_i, y_j)$, the estimated value of $u$ at $(t^n, x_i, y_j)$} \\
    \hat{\pi}^n_{ij} \text{ such that } \pi \in \hat{\lambda}       & & \sim & & & \text{$\hat{\pi}(t^n, x_i, y_j)$, the estimated value of $\pi$ at $(t^n, x_i, y_j)$.}
    \end{aligned}
```
When there is no risk of confusion, I will omit spatial indices corresponding to the consideration of a parameter over all space, as,
```math
\begin{aligned}
    u^n                                                             & & \sim & & & \text{$u(t^n)$, an array of size $I \times J$ of the true (experimental) value of $u$ at every $(x_i, y_j)$ at $t^n$} \\
    \pi^n \text{ such that } \pi \in \lambda                        & & \sim & & & \text{$\pi(t^n)$, an array of size $I \times J$ of the true (underlying) value of $\pi$ at every $(x_i, y_j)$ at $t^n$} \\
    \hat{u}^n                                                       & & \sim & & & \text{$u(t^n)$, an array of size $I \times J$ of the estimated value of $u$ at every $(x_i, y_j)$ at $t^n$} \\
    \hat{\pi}^n \text{ such that } \hat{\pi} \in \hat{\lambda}      & & \sim & & & \text{$\pi(t^n)$, an array of size $I \times J$ of the estimated value of $\pi$ at every $(x_i, y_j)$ at $t^n$.}
\end{aligned}
```
Hopefully, this naturally leads one to realize $\phi \in \{ u, \kappa, v, w, f, \hat{u}, \hat{\kappa}, \hat{v}, \hat{w}, \hat{f} \}$ is an array of size $N \times I \times J$ of the (resp: true or estimated) value of $\phi$ at every $(t^n, x_i, y_j)$.

Finally, I will notate derivatives, e.g., with respect to a parameter $s$, as,
```math
\begin{aligned}
    (u^n_{ij})_s                                                                & & \sim & & & \text{$\partial_s u(t^n, x_i, y_j)$, the $s$-partial of the true (experimental) value of $u$ at $(t^n, x_i, y_j)$} \\
    (\pi^n_{ij})_s \text{ such that } \pi \in \lambda                           & & \sim & & & \text{$\partial_s \pi(t^n, x_i, y_j)$, the $s$-partial of the true (underlying) value of $\pi$ at $(t^n, x_i, y_j)$} \\
    (\hat{u}^n_{ij})_s                                                          & & \sim & & & \text{$\partial_s \hat{u}(t^n, x_i, y_j)$, the $s$-partial of the estimated value of $u$ at $(t^n, x_i, y_j)$} \\
    (\hat{\pi}^n_{ij})_s \text{ such that } \hat{\pi} \in \hat{\lambda}         & & \sim & & & \text{$\partial_s \hat{\pi}(t^n, x_i, y_j)$, the $s$-partial of the estimated value of $\pi$ at $(t^n, x_i, y_j)$,}
\end{aligned}
```
or, considered over all space where we can omit subscripts corresponding to spatial index,
```math
\begin{aligned}
    u^n_s                                                           & & \sim & & & \text{$\partial_s u(t^n)$, an array of size $I \times J$ of the $s$-partial of the true (experimental) value of $u$ at every $(x_i, y_j)$ at $t^n$} \\
    \pi^n_s \text{ such that } \pi \in \lambda                      & & \sim & & & \text{$\partial_s \pi(t^n)$, an array of size $I \times J$ of the $s$-partial of the true (underlying) value of $\pi$ at every $(x_i, y_j)$ at $t^n$} \\
    \hat{u}^n_s                                                     & & \sim & & & \text{$\partial_s u(t^n)$, an array of size $I \times J$ of the $s$-partial of the estimated value of $u$ at every $(x_i, y_j)$ at $t^n$} \\
    \hat{\pi}^n_s \text{ such that } \hat{\pi} \in \hat{\lambda}    & & \sim & & & \text{$\partial_s \pi(t^n)$, an array of size $I \times J$ of the $s$-partial of the estimated value of $\pi$ at every $(x_i, y_j)$ at $t^n$.}
\end{aligned}
```
where each is an array of size $I \times J$.



## The Approach
### PINNs in the General Inverse Problem Context
In the inverse problem context, PINNs are generally<sup>[ha, this section does what it says on the carton]</sup> used as a black-box for (I)BVP-type problems where we learn the unknown parameters $\lambda$ of the differential operator $D$ (possibly nonlinear!) allowing us to generate a solution for the interior of the parabolic boundary (in the terminology of Evans, see 2e p.52.)  Suppose we have the problem,
```math
\begin{aligned}
    u_t &= D[u; \lambda] & &\text{in } \Gamma \\
    u &= g & &\text{on } \partial\Gamma
\end{aligned}
```
where some subset of spacetime $\Gamma \subset\mathbb{R}^n \times \mathbb{R}^+$, $u : \Gamma \rightarrow \mathbb{R}$, $\lambda$ is some (possibly space- and time-varying) collection of parameters, and the boundary condition $g : \partial\Gamma \rightarrow \mathbb{R}$.  Simply, we want to find $\lambda$.

#### Notational aside
The general problem is, in fact, more general than the exact context in which we're applying the PINN framework.  Specifically, PINNs do not require dense sampling of the solution $u$ or rectangularly discretized data, as I have described in the [Notation](#notation) section.  For the remainder of this section, we will adopt the following more general notation: let some index set $S = \{ i \}_{i=1}^{N_S}$ represent an enumeration of all sampled points for $N_S$ total samples.  Then,
```math
\begin{aligned}
    x_i
    u_i                 & & \sim & & & \text{$u(t_i, x_i, y_i)}
\end{aligned}
```
and, generally for a subset $S^* \subset S$,
```math
    u_{S^*}
```
<!-- FIXME: make sure the indexing and enumeration here is compatible with the PDE layout like 10 lines up -->


We then calculate an approximate solution using some suitable, accurate-enough integrator<sup>[[see note on integrators](#note-on-multi-stage-and-multi-step-methods)]</sup> $I$, like,
```math
\tilde{u} = I[g; \tilde{\lambda}]
```

Note that, in the preceding discussion of these (as-of-yet imaginary) machines that produce $\tilde{\lambda}$ and $\tilde{u}$, the boundary data $g$ can be replaced with any known state of $u$, effectively just translating the problem in time.  We use $g$ to clarify notation, but, generally speaking, we will use a known state of $u$, say $u(t_n)$, for $g$, from which we attempt to calculate $\tilde{u}(t_{n + 1})$.

Finally, then, there are two basic considerations: as observers (scientists, etc...) we care about the solution error.  Note, however, that as *better* observers (*better* scientists, etc...), we ought to also care about error in the governing physics encoded in the PDE itself, that is a misbalance in the left- and right-hand sides of the PDE.  This naturally leads us to consider the two residuals,
```math
    L_u = \| u - \tilde{u} \|_\Gamma
```
and
```math
    L_D = \| u_t - D[u; \tilde{\lambda}] \|_\Gamma
```
where $\| \cdot \|_\Gamma$ is used to denote a suitable norm calculated over all (known) points on the interior of $\Gamma$, $L_u$ represents the solution residual (or loss) and $L_D$ represents the differential residual (or loss.)  We calculate the necessary partials of $\tilde{u}$ similarly to calcualting $\tilde{u}$ itself using a good-enough finite-difference scheme.

Simply, then, we'll use the sum of these two measures of loss to train our PINN,
```math
L = L_u + L_D
```

We hedge on a sufficiently capable ML suite to be able to autodifferentiate this loss function and associated compositions of integrators and derivatives.

See the section on the discovery of PDE from data from [the Wikipedia page on PINNs](https://en.wikipedia.org/wiki/Physics-informed_neural_networks#Data-driven_discovery_of_partial_differential_equations), from which this notation is adapted.<sup>[[see note on terminology](#terminological-note)]</sup>

Also [see this paper](https://doi.org/10.1016/j.jcp.2018.10.045), refered to by the PINN Wikipedia page.


#### Terminological note
As far as this author is concerned, the relevant physics encoded in a PDE is not determined by the coefficients but by the differential operators that define the PDE.  Accordingly, describing this type of inverse problem as the "discovery of PDE" seems categorically incorrect or a gross aggrandizement of one's own work.

We're not discovering PDE here, we're just solving for the parameters that make the model work.  These two things are not the same.


#### Note on multi-stage and multi-step methods
Because $\kappa$, $v$, and $f$ are spatiotemporally variable, (respectively the $\tilde{\cdot}$ versions, as well,) it is unclear how (or wrong) to use a multi-stage scheme to calculate a latter timestep precisely because $\kappa$, $v$, and $f$ vary between timesteps, (at least assuming their inter-step variance is significant.  This can be disregarded if they can be assumed to have small inter-step variance.)

Rather than a multi-stage scheme, multi-step schemes can be used to integrate the PDE, relying only on the good-enough approximations for $\kappa$, $v$ and $f$.


#### Note on bootstrapping this method
It's not (yet) clear how to bootstrap this method, in particular for the use of a multi-step integration scheme.  Darn.


### The Endgame
Right now, what we want really isn't too complex:
<p align="center">
  <img src="readme-figure/complete-diagram.drawio.svg"/>
</p>

Notationally, we use $u$ to denote experimental measurements of the solution, which we assume to be exact measurements of the true solution at discrete locations, $\tilde{u}$ to denote the approximate solution calculated using estimates $\tilde{\kappa}$, $\tilde{v}$, and $\tilde{f}$ of the true PDE parameters $\kappa$, $v$, and $f$.

All that said, things get a little more complicated when we begin to understand the complexity required to train such a PINN.  Right now, this seems like a sensible loss function:
```math
    L = \left\| u(t_{n+1}) - \tilde{u}(t_{n+1}) \right\|_\Gamma + \left\| u_t(t_n) - \nabla \cdot \left( \tilde{\kappa}(t_n) \nabla u(t_n) \right) - \tilde{v}(t_n) \cdot \nabla u(t_n) - \tilde{f}(t_n) \right\|_\Gamma
```
This loss function corresponds to this information diagram:
<p align="center">
  <img src="readme-figure/complete-loss.drawio.svg"/>
</p>

Read [this section](#pinns-in-the-general-inverse-problem-context) about why this is a sensible choice.


#### Note on a small problem
This loss function does not enforce $u : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow [0, 1]$!


### Current Work
Right now, we've simplified the bigger picture and are attempting to learn (with reference to the relevant literature) the diffusion parameter $\kappa$ of a diffusion-type PDE,
```math
\begin{aligned}
    u_t &= \nabla \cdot \left( \kappa \nabla u \right) & &\text{in } \Gamma \\
    u &= g & &\text{on } \partial\Gamma
\end{aligned}
```
where $u : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$ and $\kappa: \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$.

The structure of this algorithm is shown in this figure:
<p align="center">
  <img src="readme-figure/diffusion-diagram.drawio.svg"/>
</p>

A sensible loss function, analogous to that discussed in [this section](#pinns-in-the-general-inverse-problem-context), seems to be as follows:
```math
    L = \left\| u(t_{n+1}) - \tilde{u}(t_{n+1}) \right\|_\Gamma + \left\| u_t(t_n) - \nabla \cdot \left( \tilde{\kappa}(t_n) \nabla u(t_n) \right) \right\|_\Gamma
```
which corresponds to this information diagram:
<p align="center">
  <img src="readme-figure/diffusion-loss.drawio.svg"/>
</p>


## Diagrams
There is no standard for diagrams (except possibly Microsoft Visio, which is itself ideologically- and cost-prohibitive,) but, for this project, I am using [draw.io](https://draw.io/) for diagrams.


## The Software
There exist a few implementations of PINNs in various ML toolkits.  
1. Raissi has [the original TensorFlow implementation](https://github.com/maziarraissi/PINNs) on Github that was used to produce the results for [the 2019 paper.](https://www.sciencedirect.com/science/article/abs/pii/S0021999118307125)
2. [An updated repository](https://github.com/rezaakb/pinns-torch) built with PyTorch exists and provides probably a lot of the functionality I'm going to reimplement here.

We're trying something related but different enough to warrant a unique/our own implementation.

The author suggests users of this software use Anaconda/conda to maintain the python environment in which this code operates, but you should use whatever works and what you know best.

### Conda
The following commands should Just Work™ with conda to create an environment called neuralpde in which this code will run.  Begin by creating the environment,
```shell
conda create -n neuralpde python=3.11 ipython scipy numpy matplotlib jupyter jupyterlab tqdm basemap basemap-data-hires netcdf4 -c conda-forge
```
You must also install a suitable version of PyTorch.  This was previously possible with conda, but the PyTorch collaboration ceased its official support for the platform, so PyPI/pip is the only remaining convenient way of doing so.

Be sure to activate the new environment you just created.  In this tutorial, that is probably with the command `conda activate neuralpde`.

Go to [pytorch.org](https://pytorch.org/), scroll down, and select a suitable version of PyTorch for your machine.  With CUDA, the command you need is most likely,
```shell
conda activate neuralpde
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
```
Without CUDA, it is most likely,
```shell
conda activate neuralpde
pip install torch torchvision torchaudio
```

### Not Conda
Uhhh, somehow install these packages:
```
ipython
scipy
numpy
matplotlib
jupyter
jupyterlab
tqdm
basemap
basemap-data-hires
netcdf4
torch
torchvision
torchaudio
```
where you install the most appropriate versions of the `torch*` packages for your system.


## Sea Ice Data
### Version 4
See [this](https://nsidc.org/data/g02202/versions/4) for more information on NOAA/NSIDC sea ice concentration data format 4.  In particular, the user manual is of significant aid.

#### Downloading Data Files
Download data files from [this link](https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/aggregate/) (note that this link can also be found from the NOAA/NSIDC landing page, above.)  A tool like wget can be of particular aid.  From the project root, run something like the following command:
```shell
mkdir -p data/V4/
cd data/V4/
wget --recursive --no-parent --no-host-directories --cut-dirs 4 --timestamping --execute robots=off --accept *daily*.nc https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/aggregate/
```

### Version 5
See [this](https://nsidc.org/data/g02202/versions/5) for more information on NOAA/NSIDC sea ice concentration data format 5.  In particular, the user manual is of significant aid.

#### Downloading Data Files
Download data files from [this link](https://noaadata.apps.nsidc.org/NOAA/G02202_V4/north/aggregate/) (note that this link can also be found from the NOAA/NSIDC landing page, above.)  A tool like wget can be of particular aid.  From the project root, run something like the following command:
```shell
mkdir -p data/V5/
cd data/V5/
wget --recursive --no-parent --no-host-directories --cut-dirs 4 --timestamping --execute robots=off --accept *daily*.nc https://noaadata.apps.nsidc.org/NOAA/G02202_V5/north/aggregate/
```
