"""Reconstruct a 3D structure from its distance matrix, visualize, and log it.

Usage:
    python run.py                          # synthetic protein, full distances
    python run.py --mode contact           # harder: contact-map only
    python run.py --pdb 1UBQ               # a real structure (downloaded from RCSB)
    python run.py --pdb 1UBQ --chain A

Every run writes into a results/ folder:
    results/plots/loss_curve_<tag>.png       -- training loss vs step
    results/plots/reconstruction_<tag>.png   -- ground truth vs recovered backbone
    results/data/results.csv                 -- ONE appended row per run
where <tag> is e.g. "1UBQ_A_full" or "synthetic_n64_contact".

The CSV accumulates across runs, so you can sort/compare how steps, lr, cutoff,
etc. change the final loss and RMSD.
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os

import numpy as np

import data # msking and fetching
import distances as dg # main jax algo, here i have to do the very scary human learning

RESULTS_DIR = "results"
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
DATA_DIR = os.path.join(RESULTS_DIR, "data")
CSV_PATH = os.path.join(DATA_DIR, "results.csv")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdb", default=None,
                    help="PDB id (e.g. 1UBQ) or path to a .pdb file; "
                         "an id is downloaded from RCSB if not already local")
    ap.add_argument("--chain", default=None, help="chain id to select from the PDB")
    ap.add_argument("--mode", choices=["full", "contact"], default="full",
                    help="use all distances (full) or only contacts")
    ap.add_argument("--n", type=int, default=64, help="length of synthetic protein")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--cutoff", type=float, default=8.0, help="contact cutoff (A)")
    args = ap.parse_args()

    # --- ground truth ------------------------------------------------------
    if args.pdb:
        pdb_path = data.ensure_pdb(args.pdb)          # downloads if not present
        true_coords = data.load_pdb_ca(pdb_path, chain=args.chain)
        source = os.path.splitext(os.path.basename(pdb_path))[0]   # e.g. "1UBQ"
        tag = source + (f"_{args.chain}" if args.chain else "")
        print(f"Loaded {len(true_coords)} C-alpha atoms from {pdb_path}")
    else:
        true_coords = data.synthetic_protein(n=args.n)
        source = "synthetic"
        tag = f"synthetic_n{args.n}"
        print(f"Generated synthetic protein with {len(true_coords)} residues")

    n = len(true_coords)
    D = data.distance_matrix(true_coords)
    if args.mode == "contact":
        mask = data.contact_mask(D, cutoff=args.cutoff)
        frac = mask.sum() / (n * (n - 1))
        print(f"Contact mode: observing {frac:.1%} of pairwise distances")
    else:
        mask = data.full_mask(n)
        print("Full mode: observing all pairwise distances")

    # --- solve -------------------------------------------------------------
    print("Solving...")
    X, losses, _ = dg.solve(D, mask, n_steps=args.steps, lr=args.lr)

    aligned, rmsd, mirrored = dg.best_rmsd(X, true_coords)
    print(f"\nFinal loss : {losses[-1]:.4f}")
    print(f"RMSD to truth : {rmsd:.3f} A" + ("  (mirror image)" if mirrored else ""))

    # --- save plots + log to CSV ------------------------------------------
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

    # Encode the knobs in the tag so different configs get distinct plot files
    # (the CSV logs every run regardless; this keeps the PNGs from overwriting).
    tag += f"_{args.mode}_s{args.steps}_lr{args.lr:g}"
    if args.mode == "contact":
        tag += f"_cut{args.cutoff:g}"
    loss_png = os.path.join(PLOTS_DIR, f"loss_curve_{tag}.png")
    recon_png = os.path.join(PLOTS_DIR, f"reconstruction_{tag}.png")

    plotted = False
    try:
        _plot(losses, aligned, true_coords, rmsd, loss_png, recon_png)
        plotted = True
        print(f"Saved plots -> {loss_png}\n              {recon_png}")
    except Exception as e:  # matplotlib optional / headless issues
        print(f"(skipped plots: {e})")

    _log_csv(CSV_PATH, {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "tag": tag,
        "source": source,
        "n_residues": n,
        "mode": args.mode,
        "steps": args.steps,
        "lr": args.lr,
        "cutoff": args.cutoff if args.mode == "contact" else "",
        "chain": args.chain or "",
        "final_loss": round(float(losses[-1]), 6),
        "rmsd": round(float(rmsd), 4),
        "mirror_image": mirrored,
        "loss_png": loss_png if plotted else "",
        "recon_png": recon_png if plotted else "",
    })
    print(f"Logged run  -> {CSV_PATH}")


def _log_csv(path: str, row: dict) -> None:
    """Append one run as a row, writing the header only if the file is new."""
    new_file = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _plot(losses, recovered, truth, rmsd, loss_png, recon_png) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d proj)

    fig = plt.figure(figsize=(5, 4))
    plt.semilogy(losses)
    plt.xlabel("step")
    plt.ylabel("stress loss (log)")
    plt.title("Reconstruction loss")
    plt.tight_layout()
    fig.savefig(loss_png, dpi=130)
    plt.close(fig)

    fig = plt.figure(figsize=(6, 5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(*truth.T, "-o", ms=3, lw=1.5, label="ground truth", color="#1f77b4")
    ax.plot(*recovered.T, "-o", ms=3, lw=1.5, label=f"recovered (RMSD {rmsd:.2f} A)",
            color="#d62728", alpha=0.8)
    ax.set_title("Backbone: truth vs reconstruction")
    ax.legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(recon_png, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    main()