"""Extension roadmap: from a distance solver toward AlphaFold's structure module.

These are deliberately left as guided TODO stubs -- doing them yourself is the
learning. They are ordered from easiest to hardest, and each one introduces a
new JAX transformation or an AlphaFold concept you read in the real source.

Run `pytest -q extensions.py` once you implement them (asserts included).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import distance_geometry as dg


# ===========================================================================
# EXTENSION 1 -- Batch many proteins at once with jax.vmap
# ---------------------------------------------------------------------------
# `stress_loss` is written for ONE structure. AlphaFold's FAPE loss is written
# for one frame and then vmapped over the whole trajectory (see folding.py:
#   fape_loss_fn = jax.vmap(fape_loss_fn, (0, None, None, 0, None, None)) ).
# Reproduce that pattern: given a batch of coordinate sets and one shared
# target, compute all losses without a Python loop.
def batched_stress_loss(Xs, target_D, mask):
    """Xs: (B, n, 3). Return (B,) losses. Implement with jax.vmap over axis 0.

    TODO: replace the loop below with
        return jax.vmap(dg.stress_loss, in_axes=(0, None, None))(Xs, target_D, mask)
    and confirm the outputs match.
    """
    return jnp.stack([dg.stress_loss(X, target_D, mask) for X in Xs])  # <- naive


# ===========================================================================
# EXTENSION 2 -- Contact-map reconstruction with a smarter loss
# ---------------------------------------------------------------------------
# In `--mode contact` you only know short-range distances. Pure stress often
# collapses long-range geometry. Add a hinge/repulsion term so non-contacting
# residues are merely pushed apart rather than pinned:
#     loss = observed-distance MSE  +  w * relu(cutoff - dist_ij)^2 over NON-contacts
# This is conceptually AlphaFold's `between_residue_clash_loss`.
def contact_loss(X, target_D, contact_mask, cutoff=8.0, repel_w=0.1):
    """TODO: implement the two-term loss described above."""
    raise NotImplementedError


# ===========================================================================
# EXTENSION 3 -- Swap in AlphaFold's FAPE loss
# ---------------------------------------------------------------------------
# FAPE = Frame Aligned Point Error. Instead of comparing a distance matrix,
# you express each residue as a local frame (rotation R_i, translation t_i),
# map predicted points into every residue's local frame, and measure the L1
# error to the ground-truth points in that same local frame. It is invariant
# to global pose but -- unlike a distance matrix -- NOT invariant to reflection,
# so it fixes the chirality ambiguity you saw in `best_rmsd`.
#
# Reference: Jumper et al. 2021, Alg. 28; and all_atom.frame_aligned_point_error
# in the AlphaFold repo.
def fape_loss(pred_frames_R, pred_frames_t, pred_points,
              true_frames_R, true_frames_t, true_points,
              clamp=10.0, eps=1e-4):
    """TODO: implement FAPE.

    Sketch:
        # bring points into each residue's local frame:
        #   x_local[i, j] = R_i^T @ (point_j - t_i)
        # do this for both prediction and truth, then
        #   err = || pred_local - true_local ||  (per i,j)
        #   err = minimum(err, clamp)
        #   return mean(err) / length_scale
    """
    raise NotImplementedError


# ===========================================================================
# EXTENSION 4 -- Invariant Point Attention (the real module you read)
# ---------------------------------------------------------------------------
# Reimplement a minimal IPA layer (folding.py: InvariantPointAttention):
#   * each residue emits query/key POINTS in its local frame
#   * project them to the global frame via the residue's affine
#   * attention logits come from the SQUARED DISTANCE between query and key
#     points (not dot products): attn_qk_point = -0.5 * sum(w * ||q - k||^2)
#   * combine with a standard scalar-attention term and a pair bias
# Then stack a few IPA layers, each predicting an update to the residue frames,
# and train them with the FAPE loss from Extension 3. That is, in miniature,
# AlphaFold's structure module.
class MiniIPA:
    """TODO: implement __call__(inputs_1d, inputs_2d, frames) -> updated features."""


if __name__ == "__main__":
    # Smoke test for Extension 1 once you switch to vmap.
    key = jax.random.PRNGKey(0)
    Xs = jax.random.normal(key, (4, 10, 3))
    D = dg.pairwise_distances(Xs[0])
    mask = jnp.ones((10, 10)).at[jnp.diag_indices(10)].set(0.0)
    print("batched losses:", batched_stress_loss(Xs, D, mask))