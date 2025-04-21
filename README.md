# NeuralPDE for Sea Ice
Research code for solving the forced advection-diffusion inverse problem for polar sea ice concentration.


#### Note on This Document
While it is my hope that this summary of the methods and current work can always be complete, correct, and concise, it is the case that this page serves as a place to record ideas and thoughts on how to proceed, which, by their nature, can virtually never satisfies those desirable qualities.

If you spot any egregious errors, please open a bug with the details or your questions or a pull request with the correction.


#### Note on Notation
I've attempted to maintain consistent notation throughout.  However, in various places certain notations are more or less useful, and I possess not the foresight to use such notation throughout or it was unnecessarily cumbersome to use everywhere.  In these places, I have used the notation I thought to minimize ambiguity about the topics discussed.


## The Problem
We're trying to understand ice flows in Earth's polar regions.  NOAA and NASA publish large sea ice concentration datasets dating back nearly 50 years.

From observation, we believe the relevant physics to be governed by a forced advection-diffusion equation, like
```math
    u_t = \nabla \cdot \left( \kappa \nabla u \right) + \vec{v} \cdot \nabla u + f
```
where space $x \in \mathbb{R}^2$, time $t \in \mathbb{R}^+$, the diffusion field $\kappa : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$, the velocity field $(v, w)^T =: \vec{v} : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}^2$, the forcing term $f : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow \mathbb{R}$ corresponding to ice freeze and thaw, and sea ice concentration $u : \mathbb{R}^2 \times \mathbb{R}^+ \rightarrow [0, 1]$.

The problem may be called an *anisotropic forced advection-diffusion problem.*


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
    \kappa, & \; \hat{\kappa}                                          & \sim & & & \text{diffusivity (resp. true and estimated)} \\
    v, & \; \hat{v}                                                    & \sim & & & \text{first component of velocity (resp. true and estimated)} \\
    w, & \; \hat{w}                                                    & \sim & & & \text{second component of velocity (resp. true and estimated)} \\
    f, & \; \hat{f}                                                    & \sim & & & \text{forcing (resp. true and estimated.)} \\
\end{aligned}
```

Then, I will interpret each of the following parameters according to,
```math
\begin{aligned}
    t^n                                                             & & \sim & & & n^{th} \text{ timestep, } n \in [1, \dots, N] \\
    x_i                                                             & & \sim & & & i^{th} \text{ } x \text{ coordinate (position), } i \in [1, \dots, I] \\
    y_j                                                             & & \sim & & & j^{th} \text{ } y \text{ coordinate (position), } j \in [1, \dots, J] \\
    u^n_{ij}                                                        & & \sim & & & u(t^n, x_i, y_j) \text{, the true (experimental) value of } u \text{ at } (t^n, x_i, y_j) \\
    \pi^n_{ij} \text{ such that } \pi \in \lambda                   & & \sim & & & \pi(t^n, x_i, y_j) \text{, the true (underlying) value of } \pi \text{ at } (t^n, x_i, y_j) \\
    \hat{u}^n_{ij}                                                  & & \sim & & & \hat{u}(t^n, x_i, y_j) \text{, the estimated value of } u \text{ at } (t^n, x_i, y_j) \\
    \hat{\pi}^n_{ij} \text{ such that } \pi \in \hat{\lambda}       & & \sim & & & \hat{\pi}(t^n, x_i, y_j) \text{, the estimated value of } \pi \text{ at } (t^n, x_i, y_j).
\end{aligned}
```
When there is no risk of confusion, I will omit spatial indices corresponding to the consideration of a parameter over all space, as,
```math
\begin{aligned}
    u^n                                                             & & \sim & & & u(t^n) \text{, an array of size } I \times J \text{ of the true (experimental) value of } u \text{ at every } (x_i, y_j) \text{ at } t^n \\
    \pi^n \text{ such that } \pi \in \lambda                        & & \sim & & & \pi(t^n) \text{, an array of size } I \times J \text{ of the true (underlying) value of } \pi \text{ at every } (x_i, y_j) \text{ at } t^n \\
    \hat{u}^n                                                       & & \sim & & & \hat{u}(t^n) \text{, an array of size } I \times J \text{ of the estimated value of } u \text{ at every } (x_i, y_j) \text{ at } t^n \\
    \hat{\pi}^n \text{ such that } \hat{\pi} \in \hat{\lambda}      & & \sim & & & \hat{\pi}(t^n) \text{, an array of size } I \times J \text{ of the estimated value of } \pi \text{ at every } (x_i, y_j) \text{ at } t^n.
\end{aligned}
```
Hopefully, this naturally leads one to realize $\phi \in (u, \kappa, v, w, f, \hat{u}, \hat{\kappa}, \hat{v}, \hat{w}, \hat{f})$ is an array of size $N \times I \times J$ of the (resp: true or estimated) value of $\phi$ at every $(t^n, x_i, y_j)$.

Finally, I will notate derivatives, e.g., with respect to a parameter $\chi$, as,
```math
\begin{aligned}
    (u^n_{ij})_\chi                                                                & & \sim & & & \partial_\chi u(t^n, x_i, y_j) \text{, the } \chi \text{-partial of the true (experimental) value of } u \text{ at } (t^n, x_i, y_j) \\
    (\pi^n_{ij})_\chi \text{ such that } \pi \in \lambda                           & & \sim & & & \partial_\chi \pi(t^n, x_i, y_j) \text{, the } \chi \text{-partial of the true (underlying) value of } \pi \text{ at } (t^n, x_i, y_j) \\
    (\hat{u}^n_{ij})_\chi                                                          & & \sim & & & \partial_\chi \hat{u}(t^n, x_i, y_j) \text{, the } \chi \text{-partial of the estimated value of } u \text{ at } (t^n, x_i, y_j) \\
    (\hat{\pi}^n_{ij})_\chi \text{ such that } \hat{\pi} \in \hat{\lambda}         & & \sim & & & \partial_\chi \hat{\pi}(t^n, x_i, y_j) \text{, the } \chi \text{-partial of the estimated value of } \pi \text{ at } (t^n, x_i, y_j),
\end{aligned}
```
or, considered over all space where we can omit subscripts corresponding to spatial index,
```math
\begin{aligned}
    u^n_\chi                                                           & & \sim & & & \partial_\chi u(t^n) \text{, an array of size } I \times J \text{ of the } \chi \text{-partial of the true (experimental) value of } u \text{ at every } (x_i, y_j) \text{ at } t^n \\
    \pi^n_\chi \text{ such that } \pi \in \lambda                      & & \sim & & & \partial_\chi \pi(t^n) \text{, an array of size } I \times J \text{ of the } \chi \text{-partial of the true (underlying) value of } \pi \text{ at every } (x_i, y_j) \text{ at } t^n \\
    \hat{u}^n_\chi                                                     & & \sim & & & \partial_\chi u(t^n) \text{, an array of size } I \times J \text{ of the } \chi \text{-partial of the estimated value of } u \text{ at every } (x_i, y_j) \text{ at } t^n \\
    \hat{\pi}^n_\chi \text{ such that } \hat{\pi} \in \hat{\lambda}    & & \sim & & & \partial_\chi \pi(t^n) \text{, an array of size } I \times J \text{ of the } \chi \text{-partial of the estimated value of } \pi \text{ at every } (x_i, y_j) \text{ at } t^n.
\end{aligned}
```
where each is an array of size $I \times J$.


## The Approach
### PINNs in the General Inverse Problem Context
For inverse problems, PINNs are generally constructed as solution operators for a PDE with unknown parameterization.  In particular, we learn *both* the solution and unknown parameters $\lambda$ of the (possibly nonlinear) differential operator $D$.  Then, by construction, the PINN is itself a solution to the PDE for the interior of the parabolic boundary (in the terminology of Evans, see 2e p.52.)

Suppose we have the problem,
```math
\begin{aligned}
    u_t - D[u; \lambda] &= 0 & &\text{in } \Gamma \\
    u &= g & &\text{on } \partial\Gamma
\end{aligned}
```
where $\Gamma \subset\mathbb{R}^N \times \mathbb{R}^+$ some open subset of spacetime, the solution $u : \overline{\Gamma} \rightarrow \mathbb{R}$, parameterization $\lambda : \overline{\Gamma} \rightarrow \mathbb{R}^{N_\lambda}$ is some collection of parameters, and the boundary condition $g : \partial\Gamma \rightarrow \mathbb{R}$.


#### Notational aside
The general problem is, in fact, more general than the exact context to which we're applying the PINN framework for sea ice.  Specifically, PINNs do not require dense sampling of the solution $u$ or rectangularly discretized data, as I have (implicitly) described in the [Notation](#notation) section.  For the remainder of this section, we will adopt the following more general notation: let some index set $S = \{ s \}_{s=1}^{N_S}$ represent an enumeration of all sampled points for $N_S$ total samples.  Then,
```math
\begin{aligned}
    t^s             & & \sim & & & \text{the time corresponding to the } s^{th} \text{ sample} \\
    x^s             & & \sim & & & \text{the location corresponding to the } s^{th} \text{ sample like } ( x^s_n )_{n=1}^N =: x^s \in \mathbb{R}^N \\
    u^s             & & \sim & & & u(t^s, x^s) \text{, the true (experimental) value of } u \text{ at } (t^s, x^s) \\
    \lambda^s       & & \sim & & & \lambda(t^s, x^s) \text{, the true (underlying) value of } \lambda \text{ at } (t^s, x^s) \text{ like } (\lambda_n)_{n=1}^{N_\lambda} =: \lambda^s \in \mathbb{R}^{N_\lambda},
\end{aligned}
```
and, generally for a subset $S^* \subset S$,
```math
\begin{aligned}
    t^{S^*}         & & \sim & & & \{ t^s \}_{s \in S^*} \text{, the time corresponding to the } s^{th} \text{ sample at every } s \in S^* \\
    x^{S^*}         & & \sim & & & \{ x^s \}_{s \in S^*} \text{, the location corresponding to the } s^{th} \text{ sample at every } s \in S^* \\
    u^{S^*}         & & \sim & & & \{u(t^s, x^s)\}_{s \in S^*} \text{, the true (experimental) value of } u \text{ at } (t^s, x^s) \text{ at every } s \in S^* \\
    \lambda^{S^*}   & & \sim & & & \{\lambda(t^s, x^s)\}_{s \in S^*} \text{, the true (underlying) value of } \lambda \text{ at } (t^s, x^s) \text{ at every } s \in S^*.
\end{aligned}
```
As in the former discussion of the application of PINNs to the sea ice problem, for any quantity $\phi \in (u, \lambda)$, I will represent its estimate as $\hat{\phi}$.  Similarly, for any quantity $\phi \in (u, \lambda, \hat{u}, \hat{\lambda})$, I will notate its derivative with respect to a parameter $\chi$ as $\phi_\chi := \partial_\chi \phi$. 

There are at least two important subsets of $S$,
```math
\begin{aligned}
    S^\Gamma                & & \sim & & & \text{the set of sample points in } \Gamma \\
    S^{\partial\Gamma}      & & \sim & & & \text{the set of sample points on } \partial\Gamma.
\end{aligned}
```
We now return to the consideration of PINNs for the general inverse PDE problem.

Summarily: we have $N_S$ sampled points $(t^S, x^S, u^S)$, and we want to find approximations for $u(t, x)$ and $\lambda(t, x)$.

We approximate $u(t, x)$ and $\lambda(t, x)$ with a *physics-informed neural network* $P$ as
```math
\left[ \hat{u}(t, x), \hat{\lambda}(t, x) \right] = P(t, x).
```

Conditioned on the complexity of the network and *not* subject to additional assumptions (e.g., about the PDE or solution,) the PINN $P$ reproduces optimal approximations $\hat{u}(t, x)$ and $\hat{\lambda}(t, x)$ to $u(t, x)$ and $\lambda(t, x)$, respectively, by minimizing the loss,
```math
\begin{aligned}
    L &= \biggl\| \hat{u}^s - u^s \biggr\|_{s \in S} + \biggl\| \hat{u}_t^s -  D\left[\hat{u}^s; \hat{\lambda}^s\right] \biggr\|_{s \in S^\Gamma} \\
    &= \biggl\| \hat{u}^{S} - u^{S} \biggr\| + \biggl\| \hat{u}_t^{S^\Gamma} - D\left[\hat{u}^{S^\Gamma}; \hat{\lambda}^{S^\Gamma}\right] \biggr\|
\end{aligned}
```
where $\| \cdot \|$ is some suitable notion of norm.

A **crucial realization** is that derivatives $\hat{u}_t$ and combinations of derivatives $D[\hat{u}; \hat{\lambda}]$ can be computed by applying the chain rule for compositions of functions by autodifferentiation of the PINN.

The loss $L$ is minimized in $K$ iterations on a subset of the data $S^k \subseteq S$ using a gradient descent algorithm, (e.g., BFGS and its variants, [stochastic gradient descent](https://www.cs.toronto.edu/%7Ehinton/absps/momentum.pdf), [Adam](https://arxiv.org/abs/1412.6980), etc.)  For each iteration $k$, $S^k$ is (*or can be*) randomly selected from $S$ depending on computational limitations (e.g., if the dataset is too large to fit in a computer's memory or if the complexity of the PINN is such that memory requirements for its weights do not allow simultaneous training on multiple data.)  For small problems, $S^k$ can be chosen such that $S^k = S$ for every $k$.

See [Raissi et al. 2019](https://doi.org/10.1016/j.jcp.2018.10.045) for more details, on which the above is significantly based.  The [Wikipedia page on PINNs](https://en.wikipedia.org/wiki/Physics-informed_neural_networks) also serves as an excellent resource and encyclopedia for additional resources.


### PINNs for the Constant Scalar-Parameterized Inverse Problem
Raissi et al. 2019, in addition to other interesting discussion, describes how to apply the PINN framework to problems with data collected at sparse time steps.  In particular, this can be thought of as a specialization of the preceeding discussion on PINNs in the general inverse problem context.  The following discussion covers section 4.2 of Raissi et al. 2019.

Suppose we have some collection of samples at a time $t^n$ and time $t^{n+1}$.   In the notation of the preceding section, we have $N_S = N^{n}_S + N^{n+1}_S$ samples corresponding to index set $S = S^n \cup S^{n+1}$ such that,
```math
\begin{aligned}
    t^{S^n} &= t^n \\
    t^{S^{n+1}} &= t^{n+1}.
\end{aligned}
```
Note that, of course, this requires $S^n \cap S^{n+1} = \varnothing$.

Accordingly, we have coordinates and measurements, where
```math
\begin{aligned}
    x^{S^n} & & &\text{and} & x^{S^{n+1}} & & &\text{in space } \mathbb{R}^N, & \text{and} \\
    u^{S^n} & & &\text{and} & u^{S^{n+1}} & & &\text{are scalars.}
\end{aligned}
```
Let some PDE in consideration,
```math
u_t = D[u; \lambda].
```

Raissi develops this framework assuming spatiotemporally constant differential parameterization $\lambda$.  While the discussion develops assuming spatially-constant $\lambda$ for conveience, assuming temporally-constant $\lambda$ is generally necessary to guarantee uniqueness of $\lambda$, (though it is unclear if this is both necessary and sufficient.  See [this section](#uniqueness-of) for an example of why this is the case and when non-unique $\lambda$ can be avoided.)  However, [the general framework above](#pinns-in-the-general-inverse-problem-context) theoretically (though its analysis has not yet been completed) permits extraction of temporally- and even spatio-temporally varying parameterization $\lambda$.

Recall the general form of a Runge-Kutta numerical integration scheme: momentarily, assume that we have knowledge of $u$ at each of the $q$ Runge-Kutta stages, $u^{n + c_1}, u^{n + c_2}, \dots, u^{n + c_q}$, and some value of $\lambda$.  Note that, generally speaking, these values are hidden.  An RK scheme of $q$ stages for the PDE above integrating with time step $\Delta t$ can be expressed as,
```math
\begin{aligned}
    u^{n+1} &= u^n + \Delta t \sum_{i=1}^q b_i k_i \\
    \text{where} \quad k_i &= D\left[ u^n + \Delta t \sum_{j=1}^q a_{ij} k_j; \lambda \right].
\end{aligned}
```
This is equivalent to,
```math
\begin{aligned}
    u^{n+c_i} &= u^n + \Delta t \sum_{j=1}^q a_{ij} D\left[ u^{n + c_j}; \lambda \right] \\
    u^{n+1} &= u^n + \Delta t \sum_{j=1}^q b_j D\left[ u^{n + c_j}; \lambda \right].
\end{aligned}
```
Raissi demonstrates how to rearrange the previous equations to estimate the solution $u$ at the endpoints of the interval from the predicted intermediate solutions $u^{n + c_j}$ and parameters $\lambda$ as below.  Specifically, the first equation below is the first equation above reordered, and the second equation below can be found by subtracting the equations above.
```math
\begin{aligned}
    u^n_i &= u^{n+c_i} - \Delta t \sum_{j=1}^q a_{ij} D\left[ u^{n + c_j}; \lambda \right] \\
    u^{n+1}_i &= u^{n+c_i} - \Delta t \sum_{j=1}^q (a_{ij} - b_j) D\left[ u^{n + c_j}; \lambda \right]
\end{aligned}
```
for $i = 1, \dots, q$.  Note that Raissi formulates the PDE such that $D[u; \lambda]$ has the opposite sign, which results in an opposite sign associated with the $\Delta t \sum (\cdots)$ term.  

We construct a PINN $P$ to predict the values $u^{n + c_1}, u^{n + c_2}, \dots, u^{n + c_q}$ and $\lambda$ for all $x \in x^S = x^{S^n} \cup x^{S^{n+1}}$,
```math
\left[ \hat{\lambda}, \hat{u}^{n + c_1}, \hat{u}^{n + c_2}, \dots, \hat{u}^{n + c_1} \right] = P(x^S)
```
Specfically, our PINN has yielded estimates of the solution $\hat{u}^{n + c_1}(x^{S^n}), \hat{u}^{n + c_2}(x^{S^n}), \dots, \hat{u}^{n + c_1}(x^{S^n})$, from which we estimate the solution at each $(t^{S^n}, x^{S^n})$ using the equation above for $u^{n}_i$, and $\hat{u}^{n + c_1}(x^{S^{n+1}}), \hat{u}^{n + c_2}(x^{S^{n+1}}), \dots, \hat{u}^{n + c_1}(x^{S^{n+1}})$, from which we estimate the solution at each $(t^{S^{n+1}}, x^{S^{n+1}})$ using the equation above for $u^{n+1}_i$.

Finally then, we compute a loss as some suitable norm of the errors of these estimates for the solution at each $x^{S^n}$ and $x^{S^{n+1}}$,
```math
\begin{aligned}
    L &= L^{n}_i + L^{n+1}_i, & &\text{where} \\
    L^{n}_i &= \left\| \hat{u}^{n}_i(x^{S^n}) - u^{S^n} \right\| & &\text{and} \\
    L^{n+1}_i &= \left\| \hat{u}^{n+1}_i(x^{S^{n+1}}) - u^{S^{n+1}} \right\|
\end{aligned}
```
for $i = 1, \dots, q$.


#### Uniqueness of $\lambda$
As a simple example of why it is necessary to assume $\lambda$ is constant on the interval $(t^n, t^{n+1})$ is to consider the advection equation,
```math
\begin{aligned}
    u_t + \lambda u_x &= 0  & &\text{in } \mathbb{R}^+ \times \mathbb{R} \\
    u(0, x) &= u_0(x)       & &\text{on } \{ t = 0 \} \times \mathbb{R}.
\end{aligned}
```
Recall the solution to this problem is given by,
```math
u(t, x) = u_0(x - \lambda t).
```

Assume $\lambda$ is not constant on the interval $(t^n, t^{n+1})$.  Then there exist infinite time-varying parameterization $\lambda$ of the PDE.  In particular, the following two parameterizations produce data recorded at the endpoints $t^n$ and $t^{n+1}$,
```math
\begin{aligned}
    \lambda(t) &= \lambda_0 & &\text{or} & \lambda(t) &= \begin{cases} \phantom{-}3 \lambda_0 & t^n < t < \frac{2}{3} \Delta t \\ -3 \lambda_0 & t\phantom{^n} \geq \frac{2}{3} \Delta t \end{cases}
\end{aligned}
```
for $\Delta t = t^{n+1} - t^n$.

Exactly, this produces,
```math
\begin{aligned}
    u(t, x) &= u_0(x - \lambda_0 t^n) & &\text{and} & u(t, x) &= u_0(x - \lambda_0 t^n)
\end{aligned}
```
at endpoint $t^n$, and
```math
\begin{aligned}
    u(t^{n+1}, x) &= u_0(x - \lambda_0 t^{n+1}) & &\text{and} & u(t^{n+1}, x) &= u_0(x - \lambda_0 t^n - 3 \lambda_0 \frac{2}{3}\Delta t + 3 \lambda_0 \frac{1}{3}\Delta t) \\
                                                                     & & & & &= u_0(x - \lambda_0 t^n - \lambda_0 \Delta t) \\
                                                                     & & & & &= u_0(x - \lambda_0 t^{n + 1})

\end{aligned}
```
at endpoint $t^{n+1}$.

Clearly, then, allowing temporally varying $\lambda$ will not produce a unique $\lambda$ for every PDE.

There are some assumptions that can mitigate this problem.  For example, if we assume $\lambda$ is minimally temporally varying we again recover uniqueness.  This is, in fact, a relatively weak assumption related to how the data are sampled and is, generally, reasonable.


### PINNs for the Vector Field-Parameterized Forced Advection-Diffusion Inverse Problem
This section can be thought of a generalization of the ideas in the previous section, while adhering to the desire to analytically compute an accurate solution by means of a sufficiently accurate Runge-Kutta integration scheme.  Herein I discuss how I've implemented the ideas from Raissi et al. 2019 to the problem and data of forced advection-diffusion of sea ice.


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
