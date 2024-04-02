---
marp: true
title: "Optimization"
theme: uncover
paginate: true
header: "Optimization"
footer: "Grégoire Passault"
style: :root { font-size: 1.5em; }
---

<style>
@import 'bootstrap/css/bootstrap.min.css';
@import 'style.css';
</style>

# Optimization - Gradient descent
### Grégoire Passault

---

<!-- header: "Gradient descent" -->
# Gradient descent

---

## Gradient descent

Previously, we see how to solve linear systems of equations (exactly, or approximately).

<span data-marpit-fragment>

**Problem**: How to find $w$ when $f(w)$ is not linear?

</span>


---

## Gradient

Suppose we have a function $f: \mathbb{R}^n \to \mathbb{R}$.

<span data-marpit-fragment>

Let's take for example a function of two variables $x = \begin{bmatrix} x_1 \\ x_2 \end{bmatrix}$.

</span>

<span data-marpit-fragment>

In that case, the linear approximation becomes:

$$
f(x)
\approx
f(x_0)
+
\frac{df}{dx_1} \Delta x_1
+
\frac{df}{dx_2} \Delta x_2
$$

</span>

---

## Gradient

In matrix form:

$$
f(x)
\approx
f(x_0)
+
\underbrace{
\begin{bmatrix}
\frac{df}{dx_1} &&
\frac{df}{dy_2}
\end{bmatrix}
}_{\nabla f^T}
\begin{bmatrix}
\Delta x_1 \\
\Delta y_2
\end{bmatrix}
$$

We call $\nabla f$ the **gradient** of $f$.

---

## Computing the gradient

For example, let's take $f(x) = cos(x_1) + 2 x_2 x_1 - x_2^2$.

<div class="alert alert-primary">

⚙️ What is $\nabla f$, the gradient of $f$?

</div>

<span data-marpit-fragment>

$$
\nabla f
=
\begin{bmatrix}
\frac{df}{dx_1} \\
\frac{df}{dx_2}
\end{bmatrix}
=
\begin{bmatrix} -\sin(x_1) + 2 x_2 \\ 2 x_1 - 2 x_2 \end{bmatrix}
$$

</span>

---

## Gradient descent

<div class="alert alert-info">

Suppose we want to minimize $f(x)$.
**Idea**: start at $x_0$, and iteratively update $x$ by taking small steps.

</div>

<span data-marpit-fragment>

The gradient provides a local, linear approximation of f:

$$
f(x) \approx f(x_0) + \nabla f^T \Delta x
$$

It gives the direction of the **steepest** ascent (or descent) of $f$.

</span>

---

## Gradient descent

The algorithm is then:

1. Select an initial guess $x_0$.
2. Compute the gradient $\nabla f(x_0)$.
3. Update $x$ by taking a small step in the opposite direction of the gradient:
   $x_{k+1} = x_k - \alpha \nabla f(x_k)$.
4. Repeat until convergence.

<div class="alert alert-primary" data-marpit-fragment>

$\alpha$ is called the **learning rate**.

</div>

---

## Gradient descent

<center>
<video src="imgs/gradient_steps.mp4" width="800" controls />
</center>

---

## Gradient descent

<div class="alert alert-success">

We can now find the $w$ that minimizes $f(w)$, even when $f$ is not linear.

</div>

<span data-marpit-fragment>

Problem remains:
- What model $f(x, w)$ to use ?
- How to compute the gradient $\nabla f$ ?
- How to choose/adjust the learning rate $\alpha$ ?
- How to tackle over/underfitting ?

</span>

---

<!-- header: "Neural networks" -->
# Neural networks

---