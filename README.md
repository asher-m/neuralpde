# NeuralPDE for Sea Ice

Research code for solving the forced advection-diffusion inverse problem for polar sea ice concentration.


## Note on This Document
While it is my hope that this summary of the methods and current work can always be complete, correct, and concise, it is the case that this page serves as a place to record ideas and thoughts on how to proceed, which, by their nature, can virtually never satisfies those desirable qualities.

If you spot any egregious errors, please open a bug with the details or your questions or a pull request with the correction.


### Notation
I've attempted to maintain consistent notation throughout.  However, in various places certain notations are more or less useful, and I possess not the foresight to use such notation throughout or it was unnecessarily cumbersome to use everywhere.  In these places, I have used the best notation to lead to the least ambiguity about the topics discussed.


## The Problem
We're trying to understand, quantify, and classify ice flows in Earth's polar regions.  NOAA and NASA publish large sea ice concentration datasets dating back nearly 40 years.

From observation, we believe the relevant physics to be governed by a forced advection-diffusion equation, like
```math
    u_t = \nabla \cdot \left( \kappa \nabla u \right) + v \cdot \nabla u + f
```
where space $x \in \mathbb{R}^2$, time $t \in \mathbb{R}^+$, the diffusion field $\kappa : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$, the velocity field $v : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}^2$, the forcing term $f : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$ corresponding to ice freeze and thaw, and sea ice concentration $u : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow [0, 1]$.

In more words/to play with the terminology, the author posits that one could describe this problem as an "anisotropic forced advection-diffusion" problem.  Another fun way of describing this problem may be as an "inhomogeneous heterogeneous advection-diffusion" problem, (where "inhomogeneous" refers to the inhomogeneity, that is, the forcing, and "heterogeneous" refers to the spatiotemporally varying diffusion and advection.)

Note that $u$ is known (at least known on some grid in space-time.)  Accordingly, this is an inverse problem to solve for $\kappa$, $v$, and $f$.

In general, we do not know how to solve this problem.<sup>[why not?]</sup>  In this context, it is unclear if this problem is well-posed (does there exist a unique solution for each set of inputs, at least subject to reasonable assumptions?) or even well-defined (is it possible to solve this problem given the input data?)

But, like, it's probably fine.


## The Approach


### PINNs in the General Inverse Problem Context
In the inverse problem context, PINNs are generally<sup>[ha, this section does what it says on the carton]</sup> used as a black-box for (I)BVP-type problems where we learn the parameters $\lambda$ of the differential operator $D$ allowing us to generate a solution for the interior of the parabolic boundary (in the terminology of Evans, see 2e p.52.)  Suppose we have the problem,
```math
\begin{aligned}
    u_t &= D[u; \lambda] & &\text{in } \Gamma \\
    u &= g & &\text{on } \partial\Gamma
\end{aligned}
```
where $u : \mathbb{R}^n \times \mathbb{R}^+ \rightarrow \mathbb{R}$, $\lambda$ is some (possibly space- and time-varying) collection of parameters, and the boundary condition $g : \mathbb{R}^n \times \mathbb{R}^+ \rightarrow \mathbb{R}$.

Below, we will use $u$ to represent the experimentally-recorded data of the solution, $\tilde{u}$ to represent the approximate solution generated with parameters determined by the PINN, and $\lambda$ and $\tilde{\lambda}$ to represent the true- and PINN-determined parameters to the differntial operator $D$, respectively.

We assume $u$ is an exact measurement of the exact solution at the discrete positions in spacetime at which $u$ is measured.  Then, in particular, $u$ is defined only on points like,
```math
    u(x_{ij}, t_n)
```
where $t_n$ is the time at timestep $n \in [1, 2, \dots, N]$ and $x_{ij}$ is the location at $(x_i, y_j)$ for $i \in [1, 2, \dots, M_x]$ and $j \in [1, 2, \dots, M_y]$ (note the overloaded notation on $x$.)  We will omit coordinates and indices when the risk of ambiguity does not exist.

Suppose we have a PINN $P$ such that, for any boundary data $g$, $P$ yields $\tilde{\lambda}$,
```math
    \tilde{\lambda} = P[g]
```

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
