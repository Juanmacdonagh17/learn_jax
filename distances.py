"""Core differentiable distance-geometry solver in JAX.

Idea (a miniature of AlphaFold's structure module):
    coordinates are FREE PARAMETERS, and gradients flow through the
    Euclidean-distance function. Given a target (possibly partial) pairwise
    distance matrix, we recover 3D coordinates by gradient descent.

JAX concepts here:
    * jax.numpy            -- np like array ops (immutable arrays)
    * jax.value_and_grad   -- reverse mode autodiff of a scalar loss
    * jax.jit              -- compile the update step to XLA (fast)
    * optax                -- composable optimizers (Adam here)
    * pytrees              -- opt_state is a nested structure JAX walks for you
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optax


# ---------------------------------------------------------------------------
# The differentiable geometry
# ---------------------------------------------------------------------------
def pairwise_distances(X: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray: 
    # eps is a really small number so the value  of teh sq of 0 does not go to inf! its called stability epsilons apparently
    """Euclidean distance matrix for coordinates X of shape (n, 3) -> (n, n).

    The `+ eps` before sqrt is the same trick AlphaFold uses (dist_epsilon in
    InvariantPointAttention): sqrt has an infinite gradient at 0, so we nudge
    the argument away from exactly zero to keep gradients finite.
    """
    diff = X[:, None, :] - X[None, :, :]          # (n, n, 3), broadcasting, this would be a loop in "normal" py, with jax is way faster
    sq = jnp.sum(diff ** 2, axis=-1)              # (n, n), sq[i, j] = (xi−xj)² + (yi−yj)² + (zi−zj)², making the squared euclidean distance
    return jnp.sqrt(sq + eps) # eps thing (d/dx √x = 1/(2√x), which tends to Inf as x tens to 0) and the sqrt to have the actual distance
    # with real values: √(0+1e-8) = 1e-4 for example, this is what af does at the dist_epsilon in Invariant Point Attention

def stress_loss(X: jnp.ndarray, target_D: jnp.ndarray, mask: jnp.ndarray, eps: float = 1e-8) -> jnp.ndarray:
    # this f(x) is just 1,0. it calls if the predicted distance matches the target distance
    """Masked mean-squared error between predicted and target distances.

    Args:
        X:        (n, 3) predicted coordinates (the free parameters). 
        target_D: (n, n) target distances.
        mask:     (n, n) 1.0 where a distance is observed / should be matched,
                  0.0 otherwise (unobserved contacts, or the diagonal).

    Note: this loss is invariant to global rotation, translation AND
    reflection of X. 
    """
    pred_D = pairwise_distances(X) # this calculates the distance of a pair with the function from avobe!
    err = (pred_D - target_D) ** 2 # this is the error of the pred distance and the real distance, just an error
    return jnp.sum(mask * err) / (jnp.sum(mask) + eps) # eps is again a guard to not divide by 0, because mask is eathir 1 or 0
    # jnp.sum(mask) is the number of observed pairs, turns the sum into a mean over observed pairs, so the loss scale doent depend on the # of dists

# ---------------------------------------------------------------------------
# The solver: build a jitted step, then loop
# ---------------------------------------------------------------------------
def make_step(optimizer: optax.GradientTransformation):
    """Return a jitted update step closing over `optimizer`.

    This is the idiomatic JAX pattern: the optimizer is a static constant
    captured in the closure, so `jax.jit` is happy to compile the step once.
    """
    loss_and_grad = jax.value_and_grad(stress_loss) # this gives you both the loss value and its gradient in one pas
    # calling the function avobe and the one avobe etc. 
    @jax.jit # this runs in XLA, not python, here it gets weird
    # this uses ADAM (Adaptive Moment Estimation), that's an optimizer that turns SGD into a parameter update, making it more dynmaic i guess?  
    def step(X, opt_state, target_D, mask): #the first run traces the function: "blank" run  with abstract placeholders to record the operations and compiles a fast binary; so later skips Python entirely.
        loss, grads = loss_and_grad(X, target_D, mask) # computes loss and gradient
        updates, opt_state = optimizer.update(grads, opt_state, X)
        X = optax.apply_updates(X, updates) # all the opt and updates are the actualizations that adam uses 
        return X, opt_state, loss # because JAX arrays are immutable, it returns a new array, which we bind back to X, so this is JAX style?
        # IMPORTANT, X is just (n, 3), n points in (x,y,z) positions, the point cloud
    return step     


def solve(
    target_D,
    mask,
    n_steps: int = 2000,
    lr: float = 0.05,
    seed: int = 0,
    log_every: int = 200,
    snapshot_every: int | None = None,
):
    """Recover coordinates from a target distance matrix.

    Returns:
        X        : (n, 3) recovered coordinates (jnp array)
        losses   : list[float] loss per step
        snapshots: list[(step, np.ndarray)] if snapshot_every is set, else []
    """
    import numpy as np #  numpy is here because we don't need it elsewhere and it bothers the other libs    

    target_D = jnp.asarray(target_D)
    mask = jnp.asarray(mask)
    n = target_D.shape[0] # n is the number of residues

    # Immutable, explicit RNG, no global seed in JAX.
    key = jax.random.PRNGKey(seed) # jax uses a key isntead of a seed for randomnness
    scale = 0.3 * float(jnp.mean(target_D)) # sizes that initial cloud to roughly the real extent of the molecule
    X = jax.random.normal(key, (n, 3)) * scale
    # builds adam
    optimizer = optax.adam(lr)
    opt_state = optimizer.init(X)
    step = make_step(optimizer)

    losses, snapshots = [], []
    for i in range(n_steps):
        X, opt_state, loss = step(X, opt_state, target_D, mask)
        losses.append(float(loss))
        if snapshot_every and (i % snapshot_every == 0):
            snapshots.append((i, np.asarray(X)))
        if log_every and (i % log_every == 0 or i == n_steps - 1):
            print(f"  step {i:5d}   loss = {float(loss):.4f}")
    return X, losses, snapshots


# ---------------------------------------------------------------------------
# Evaluation: superpose onto ground truth (Kabsch), mirror-aware
# ---------------------------------------------------------------------------
def _kabsch_rmsd(P, Q):
    """Optimal rotation+translation of P onto Q; returns (aligned_P, rmsd)."""
    import numpy as np

    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Pc = P - P.mean(0)
    Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))          # keep a proper rotation
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    P_aligned = Pc @ R.T + Q.mean(0)
    rmsd = float(np.sqrt(np.mean(np.sum((P_aligned - Q) ** 2, axis=1))))
    return P_aligned, rmsd


def best_rmsd(pred, true):
    """RMSD to ground truth, trying the mirror image too.

    A distance matrix is invariant under reflection, so gradient descent may
    legitimately recover the *mirror* of the true structure (wrong chirality)
    at essentially zero loss. For a fair structural comparison we report the
    better of {pred, mirrored pred}. This is a genuine lesson from the project,
    not a hack -- real folding pipelines must break chirality explicitly.
    """
    import numpy as np

    pred = np.asarray(pred)
    aligned, rmsd = _kabsch_rmsd(pred, true)
    mirror = pred.copy()
    mirror[:, 0] *= -1.0
    aligned_m, rmsd_m = _kabsch_rmsd(mirror, true)
    if rmsd_m < rmsd:
        return aligned_m, rmsd_m, True
    return aligned, rmsd, False