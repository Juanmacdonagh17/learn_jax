"""Data: a synthetic protein-like chain and a real-PDB C-alpha loader.

Everything here is plain np on purpose, data  does not need to be
differentiable, so there is no reason to pay JAX's tracing cost for it.
"""
from __future__ import annotations
import numpy as np
import os
import urllib.request
from Bio.PDB import PDBParser

# ---------------------------------------------------------------------------
# Synthetic ground truth (runs with zero downloads)
# ---------------------------------------------------------------------------
def synthetic_protein(n: int = 64, seed: int = 0) -> np.ndarray:
    """A smooth, protein-like 3D backbone: a helix plus a slow random drift.

    Returns coordinates of shape (n, 3), roughly on a C-alpha distance scale
    (~3.8 A between consecutive residues).
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 6.0 * np.pi, n)
    helix = np.stack([np.cos(t), np.sin(t), 0.3 * t], axis=1)   # (n, 3)
    drift = np.cumsum(rng.normal(scale=0.15, size=(n, 3)), axis=0)
    coords = (helix + drift) * 3.8
    return coords.astype(np.float32)


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Full (n, n) Euclidean distance matrix from coordinates."""
    diff = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diff ** 2, axis=-1)).astype(np.float32)


# ---------------------------------------------------------------------------
# Masks: which distances the solver is allowed to see ## one of the main take aways is masking: you take out information from the data
# and then see if the algo can solve it
# ---------------------------------------------------------------------------
def full_mask(n: int) -> np.ndarray:
    """Observe every off-diagonal distance (easy mode: metric MDS)."""
    m = np.ones((n, n), dtype=np.float32)
    np.fill_diagonal(m, 0.0) # this just masks the diagonal
    return m


def contact_mask(D: np.ndarray, cutoff: float = 8.0, keep_chain: bool = True) -> np.ndarray:
    """Observe only short-range 'contacts' (hard, realistic mode).

    This mimics a contact map: you only know which residues are close, plus the
    chain connectivity. Reconstructing from this is the interesting case.
    """
    m = (D < cutoff).astype(np.float32) #ok?
    if keep_chain:
        idx = np.arange(D.shape[0] - 1)
        m[idx, idx + 1] = 1.0
        m[idx + 1, idx] = 1.0
    np.fill_diagonal(m, 0.0)
    return m

# ---------------------------------------------------------------------------
# Real structures: fetch + load C-alpha coordinates from a PDB
# ---------------------------------------------------------------------------
RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
 
 
def ensure_pdb(path_or_id: str, cache_dir: str = ".") -> str:
    """Return a local path to a .pdb file, downloading from RCSB if needed.
 
    Accepts either:
      * a path to an existing .pdb file (used as-is), or
      * a 4-character PDB ID like '1UBQ' (downloaded to <cache_dir>/1UBQ.pdb
        if not already there).
 
    So `--pdb 1UBQ` and `--pdb ./1UBQ.pdb` both work.
    """
    # Already a real file on disk -> nothing to do.
    if os.path.isfile(path_or_id):
        return path_or_id
 
    # Derive a clean PDB id and a target filename.
    base = os.path.basename(path_or_id)
    pdb_id = (base[:-4] if base.lower().endswith(".pdb") else base).upper()
    target = os.path.join(cache_dir, f"{pdb_id}.pdb")
    if os.path.isfile(target):
        return target
 
    url = RCSB_URL.format(pdb_id=pdb_id)
    print(f"Downloading {pdb_id} from RCSB -> {target}")
    try:
        urllib.request.urlretrieve(url, target)
    except Exception as e:  # noqa: BLE001
        # Clean up a partial file and fail with a helpful message.
        if os.path.exists(target):
            os.remove(target)
        raise SystemExit(
            f"Could not download PDB '{pdb_id}' from {url} ({e}).\n"
            f"Check the 4-letter ID, or download the file manually. Note: very "
            f"large structures may only exist as mmCIF, not legacy .pdb."
        )
    return target
 
 
def load_pdb_ca(path: str, chain: str | None = None) -> np.ndarray:
    """Load C-alpha coordinates from a local .pdb file, shape (n, 3).


    PDB file, e.g.:  https://files.rcsb.org/download/1UBQ.pdb
    (1UBQ = ubiquitin, 76 residues -- a great small test case.)
    """

    structure = PDBParser(QUIET=True).get_structure("s", path) # finally some good old fashioned bioinfo
    coords = []
    for model in structure:
        for ch in model:
            if chain and ch.id != chain:
                continue
            for res in ch:
                if "CA" in res: # the only info are the alpha carbons here!
                    coords.append(res["CA"].coord)
        break  # first model only
    return np.asarray(coords, dtype=np.float32)
