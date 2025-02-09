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
### The Endgame
<p align="center">
  <img src="readme-figure/complete-diagram.drawio.svg"/>
</p>

At this time, it seems a sensible loss function would be as,
<!-- FIXME -- this does not seem correct -->
```math
    L = \left\| \tilde{u}(t_n) - u(t_n) \right\|_{2, \Gamma} + \left\| \partial_t \tilde{u}(t_n) - \nabla \cdot \left( \kappa(t_n) \nabla \tilde{u}(t_n) \right) - v(t_n) \cdot \nabla \tilde{u}(t_n) - f(t_n) \right\|_{2, \Gamma}
```

Read [this section](#pinns-in-the-general-inverse-problem-context) about why this is a sensible choice.

#### PINNs in the General Inverse Problem Context
In the inverse problem context, PINNs are generally<sup>[ha, this section does what it says on the carton]</sup> used as a black-box for (I)BVP-type problems where we learn the parameters $\lambda$ of the differential operator $D$ allowing us to generate a solution for the interior of the parabolic boundary (in the terminology of Evans, see 2e p.52.)  Suppose we have the problem,
```math
\begin{aligned}
    u_t &= D[u; \lambda] & &\text{in } \Gamma \\
    u &= g & &\text{on } \partial\Gamma
\end{aligned}
```
where $u : \mathbb{R}^n \times \mathbb{R}^+ \rightarrow \mathbb{R}$, $\lambda$ is some (possibly space- and time-varying) collection of parameters, and the boundary condition $g : \mathbb{R}^n \times \mathbb{R}^+ \rightarrow \mathbb{R}$.

We make some assumptions and observations about the problem and use these assumptions to inform our notation and following discussion:
- Experimentally-recorded data is an exact measurement of the exact solution to the governing PDE.
- The true parameters $\lambda$ of the differential operator are unknowable.

Below, we will use $u$ to represent the experimentally-recorded data of the solution, $\tilde{u}$ to represent the approximate solution generated with parameters determined by the PINN, and $\lambda$ and $\tilde{\lambda}$ to represent the true- and PINN-determined parameters to the differntial operator $D$, respectively.

Additionally note that, while $u$ is analogous to the exact solution of the PDE in the continuum (by assumption,) it is discrete.  In particular, $u$ is defined only on points like,
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
\tilde{u} = I[g; \lambda]
```

Finally, then, there are two basic considerations: as observers (scientists, etc...) we care about the solution error.  Note, however, that as *better* observers (*better* scientists, etc...), we ought to also care about error in the governing physics encoded in the PDE itself, that is a misbalance in the left- and right-hand sides of the PDE.  This naturally leads us to consider the two residuals,
```math
    L_u = \| \tilde{u} - u \|_\Gamma
```
and
```math
    L_D = \| \tilde{u}_t - D[\tilde{u}; \lambda] \|_\Gamma
```
where $\| \cdot \|_\Gamma$ is used to denote a suitable norm calculated over all (known) points on the interior of $\Gamma$, $L_u$ represents the solution residual (or loss) and $L_D$ represents the differential residual (or loss.)  We calculate the necessary partials of $\tilde{u}$ similarly to calcualting $\tilde{u}$ itself using a good-enough finite-difference scheme.

Simply, then, we'll use the sum of these two measures of loss to train our PINN,
```math
L = L_u + L_D
```

See section of [the Wikipedia page on PINNs](https://en.wikipedia.org/wiki/Physics-informed_neural_networks#Data-driven_discovery_of_partial_differential_equations) on the discovery of PDE from data from which this notation is adapted.<sup>[[see note on terminology](#terminology-note)]</sup>

Also [see this paper](https://doi.org/10.1016/j.jcp.2018.10.045), refered to by the PINN Wikipedia page.

##### Terminological note
As far as this author is concerned, the relevant physics encoded in a PDE is not determined by the coefficients but by the differential operators that define the PDE.  Accordingly, describing this type of inverse problem as the "discovery of PDE" seems categorically incorrect or a gross aggrandizement of one's own work.

We're not discovering PDE here, we're just solving for the parameters that make the model work.  These two things are not the same.


#### Author's note
At this point in the project, it is not clear if $\kappa$, $v$, and $f$ are calculated such that we can calculate
```math
    u(x, t_{n + 1}) = I\left[ u; \kappa, v, f \right]
```
where $I$ is some sensible integration scheme (such as RK4 or a multi-stage method<sup>[[see note](#note-on-multi-stage-and-multi-step-methods)]</sup>) and $u(x, t_{n + 1})$ is **unknown.**  Such a method would be **extrapolative.** Or, alternatively,
```math
    u(x, t_n) = I\left[ u; \kappa, v, f \right]
```
where $I$ is again some sensible integration scheme<sup>[[again see note](#note-on-multi-stage-and-multi-step-methods)]</sup> and $u(x, t_n)$ is **known.**  Such a method would be **interpolative.**

In fact, it's not actually clear that the two things above are different.  Specifically, will we attempt to use this algorithm where we *do not* know $u(x, t_{n + 1})$?  After all, are we building an interpolative ***or*** extrapolative algorithm?

##### Note on multi-stage and multi-step methods
Because $\kappa$, $v$, and $f$ are spatiotemporally variable, it is unclear how (or wrong) to use a multi-stage scheme to calculate a latter timestep precisely because $\kappa$, $v$, and $f$ vary between timesteps, (at least assuming their inter-step variance is significant.  This can be disregarded if they can be assumed to have small inter-step variance.)

Rather than a multi-stage scheme, multi-step schemes can be used to integrate the PDE, relying only on the good-enough approximations for $\kappa$, $v$ and $f$.

##### Note on bootstrapping this method
It's not clear how to bootstrap this method.  Darn.

### Current Work
Right now, we've simplified the bigger picture and are attempting to learn (with reference to the relevant literature) the diffusion parameter $\kappa$ of a diffusion-type PDE,
```math
\begin{aligned}
    u_t &= \nabla \cdot \left( \kappa \nabla u \right) & &\text{in } \Gamma \\
    u &= g & &\text{on } \partial\Gamma
\end{aligned}
```

The structure of this algorithm is shown in this figure:
<p align="center">
  <img src="readme-figure/diffusion-diagram.drawio.svg"/>
</p>

## Diagrams
There is no standard for diagrams (except possibly Microsoft Visio, which is itself ideologically- and cost-prohibitive,) but, for this project, I am using [draw.io](https://draw.io/) for diagrams.
