"""Five small image-classification datasets, loaded as NCHW float32.

Nothing downloads implicitly. ``load(name)`` reads files from the data
directory; if any are missing it raises FileNotFoundError with the exact fix
command. The only code that touches the network is the explicit CLI::

    python -m mantissa_cnn.datasets download <name|all>
    python -m mantissa_cnn.datasets list

Data directory: ``./data/<name>/`` relative to the current working directory
(so ``cnn/data/`` when run from the repo root), or the ``MANTISSA_CNN_DATA``
environment variable. The directory is gitignored — datasets are never
committed.

``load(name)`` -> (X_train, y_train, X_test, y_test): X is NCHW float32
scaled to [0, 1]; y is int32 class ids (0..9 for the ten-class image sets,
0/1 for the two-class ``atlas_calo`` showers). IDX files are parsed with
numpy and verified (magic number, dtype code, dimension consistency);
CIFAR-10 is read straight out of its tar.gz (pickle batches,
encoding='bytes' — the canonical loader from the dataset page), no
extraction step; ``atlas_calo`` reads a compact voxel cache produced at
download time (see below).

| name          | train/test    | shape       | source |
|---------------|---------------|-------------|--------|
| mnist         | 60000 / 10000 | (1, 28, 28) | LeCun et al. — ossci-datasets S3 mirror (yann.lecun.com originals are auth-walled) |
| fashion_mnist | 60000 / 10000 | (1, 28, 28) | Xiao, Rasul & Vollgraf (2017), zalandoresearch/fashion-mnist |
| kmnist        | 60000 / 10000 | (1, 28, 28) | Clanuwat et al. (2018), CODH codh.rois.ac.jp/kmnist |
| qmnist        | 60000 / 60000 | (1, 28, 28) | Yadav & Bottou (2019), facebookresearch/qmnist (test = extended 60k) |
| cifar10       | 50000 / 10000 | (3, 32, 32) | Krizhevsky (2009), cs.toronto.edu/~kriz |
| atlas_calo    | ~190k / ~48k  | (1, 24, 24) | ATLAS Collab. (2021), opendata.cern.ch record 15012, CC0 |

``atlas_calo`` is a CERN Open Data set: voxelised electromagnetic (photon,
class 0) versus hadronic (charged-pion, class 1) calorimeter showers from the
ATLAS FastCaloSim GAN training samples (SIMU-2018-04). See the extended note
above ``_ATLAS_CALO_LAYERS`` for the physics, the voxel geometry, and the
shower-image construction (azimuthally averaged layer x radius energy map).

All URLs verified fetchable 2026-07.
"""
from __future__ import annotations

import gzip
import io
import os
import pickle
import sys
import tarfile
import urllib.request
from pathlib import Path
from typing import NamedTuple, Tuple

import numpy as np

__all__ = ["DATASETS", "data_dir", "download", "download_command", "load", "subset"]

_DATA_ENV = "MANTISSA_CNN_DATA"

_IDX4 = ("train-images-idx3-ubyte.gz", "train-labels-idx1-ubyte.gz",
         "t10k-images-idx3-ubyte.gz", "t10k-labels-idx1-ubyte.gz")


class _Spec(NamedTuple):
    base_url: str
    files: Tuple[str, ...]
    note: str


DATASETS = {
    "mnist": _Spec(
        "https://ossci-datasets.s3.amazonaws.com/mnist/", _IDX4,
        "handwritten digits (LeCun, Bottou, Bengio & Haffner, 1998)"),
    "fashion_mnist": _Spec(
        "https://github.com/zalandoresearch/fashion-mnist/raw/master/data/fashion/",
        _IDX4,
        "Zalando clothing thumbnails, drop-in MNIST replacement (Xiao et al., 2017)"),
    "kmnist": _Spec(
        "http://codh.rois.ac.jp/kmnist/dataset/kmnist/", _IDX4,
        "classical-Japanese Kuzushiji characters (Clanuwat et al., 2018)"),
    "qmnist": _Spec(
        "https://raw.githubusercontent.com/facebookresearch/qmnist/master/",
        ("qmnist-train-images-idx3-ubyte.gz", "qmnist-train-labels-idx2-int.gz",
         "qmnist-test-images-idx3-ubyte.gz", "qmnist-test-labels-idx2-int.gz"),
        "MNIST reconstruction with a 60k test set (Yadav & Bottou, 2019)"),
    "cifar10": _Spec(
        "https://www.cs.toronto.edu/~kriz/",
        ("cifar-10-python.tar.gz",),
        "32x32 color images, 10 classes (Krizhevsky, 2009)"),
    "atlas_calo": _Spec(
        "https://opendata.cern.ch/record/15012/files/",
        # Files load() reads: the compact voxel cache the downloader builds
        # from the remote .tgz-of-CSV samples (see _download_atlas_calo).
        ("photon_voxels.npy", "pion_voxels.npy"),
        "ATLAS calorimeter showers, photon vs pion, 2 classes "
        "(ATLAS Collab., 2021; opendata.cern.ch record 15012, CC0)"),
}


def data_dir() -> Path:
    return Path(os.environ.get(_DATA_ENV, "data"))


def download_command(name: str) -> str:
    return f"python -m mantissa_cnn.datasets download {name}"


# -- IDX / CIFAR parsing -------------------------------------------------------

_IDX_DTYPES = {0x08: np.dtype(">u1"), 0x0B: np.dtype(">i2"),
               0x0C: np.dtype(">i4"), 0x0D: np.dtype(">f4"),
               0x0E: np.dtype(">f8")}


def _read_idx(path: Path) -> np.ndarray:
    """Parse one (gzipped) IDX file, verifying magic number and sizes."""
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as f:
        buf = f.read()
    magic = int.from_bytes(buf[:4], "big")
    code, ndim = (magic >> 8) & 0xFF, magic & 0xFF
    if magic >> 16 != 0 or code not in _IDX_DTYPES or not 1 <= ndim <= 4:
        raise ValueError(f"{path}: bad IDX magic 0x{magic:08x}")
    dims = np.frombuffer(buf, ">u4", count=ndim, offset=4).astype(np.int64)
    dt = _IDX_DTYPES[code]
    expected = 4 + 4 * ndim + int(dims.prod()) * dt.itemsize
    if len(buf) != expected:
        raise ValueError(f"{path}: size {len(buf)} != expected {expected} "
                         f"for dims {dims.tolist()}")
    return np.frombuffer(buf, dt, offset=4 + 4 * ndim).reshape(dims)


def _load_idx_pair(images_path: Path, labels_path: Path):
    X = _read_idx(images_path)
    y = _read_idx(labels_path)
    if X.ndim != 3 or X.shape[1:] != (28, 28):
        raise ValueError(f"{images_path}: expected (n, 28, 28) images, got {X.shape}")
    if y.ndim == 2:            # QMNIST idx2-int: class id is column 0
        y = y[:, 0]
    if len(y) != len(X):
        raise ValueError(f"{labels_path}: {len(y)} labels for {len(X)} images")
    return X.reshape(-1, 1, 28, 28), y.astype(np.int32)


def _load_cifar10(tar_path: Path):
    def batches(names):
        xs, ys = [], []
        with tarfile.open(tar_path, "r:gz") as tar:
            for member in names:
                f = tar.extractfile(f"cifar-10-batches-py/{member}")
                d = pickle.load(f, encoding="bytes")
                data, labels = d[b"data"], d[b"labels"]
                data = np.asarray(data)
                if data.ndim != 2 or data.shape[1] != 3072:
                    raise ValueError(f"{tar_path}:{member}: bad batch shape {data.shape}")
                xs.append(data)
                ys.extend(labels)
        X = np.concatenate(xs).reshape(-1, 3, 32, 32)
        return X, np.asarray(ys, dtype=np.int32)

    Xtr, ytr = batches([f"data_batch_{i}" for i in range(1, 6)])
    Xte, yte = batches(["test_batch"])
    return Xtr, ytr, Xte, yte


def _to_f01(X):
    """uint8 [0,255] NCHW -> C-contiguous float32 [0,1]."""
    return np.ascontiguousarray(X.astype(np.float32) / 255.0)


# -- atlas_calo: ATLAS calorimeter shower images (CERN Open Data 15012) --------
#
# Physics. The ATLAS calorimeter measures particle energy by sampling the
# shower a particle develops as it stops in the detector. A photon showers
# electromagnetically — a compact cascade that deposits its energy in the
# first few (electromagnetic) sampling layers. A charged pion showers
# hadronically — a broader cascade that punches deeper, into the later
# (hadronic) layers. Telling the two apart from shower shape is a real ATLAS
# task; here it is a two-class image problem a CNN is well suited to.
#
# Source. Record 15012, "Datasets used to train the Generative Adversarial
# Networks used in ATLFast3" (ATLAS Collaboration, 2021), CC0, DOI
# 10.7483/OPENDATA.ATLAS.UXKX.TXBN. Physics: Aad et al. (ATLAS Collab.),
# "AtlFast3: the next generation of fast simulation in ATLAS", Comput.
# Softw. Big Sci. 6, 7 (2022), arXiv:2109.02551. Each event is a single
# particle (photon or pion) simulated in |eta| in [0.20, 0.25]; its energy
# deposits are converted to local cylindrical coordinates (r, alpha) around
# the particle direction and binned into voxels per calorimeter layer. The
# per-voxel energies (MeV) are stored one event per CSV row; the two `small`
# subsets used here (photon_samples.tgz, pion_samples.tgz) span 15 discrete
# energy points from 256 MeV to 4.2 TeV, ~118k photon and ~120k pion events.
# The voxel layout below is transcribed from the record's binning.xml.
#
# Image. The CSV is flat, but the voxels carry 2-D spatial structure. We
# average over the azimuth alpha (showers are ~azimuthally symmetric on
# average; the discriminating structure is longitudinal and radial) to get,
# per layer, an energy-vs-radius profile, then regrid every layer's native
# radial bins onto one common geometric radius axis. The result is a
# (layer x radius) energy map: rows are the 24 calorimeter sampling layers
# (depth), columns are common radial bins (lateral spread). Photons light
# the top (EM) rows; pions also light the deep (hadronic) rows. Energies are
# clipped at zero (a few 1e-4 of voxels are slightly negative from noise
# subtraction) and log1p-compressed to [0, 1]. One channel: this record
# stores energy only (unlike the CMS ECAL image sets, which add a timing
# channel), and a fabricated second channel would be dishonest.
#
# Storage. The downloader parses the CSV samples once into a compact float32
# voxel cache (photon_voxels.npy, pion_voxels.npy) that load()/subset()
# memory-map; the raw voxels are kept (not the expanded images), so subset()
# reads only the rows it selects and builds images for just those — the same
# uint8-first discipline the IDX/CIFAR paths use, adapted to float voxels.

_ATLAS_CALO_LAYERS = {
    # pid: [(layer_id, r_edges, n_bin_alpha), ...] for layers with >=1 voxel,
    # in the CSV's concatenation order (ascending layer id). From binning.xml.
    22: [   # photon (368 voxels: 8 + 16*10 + 19*10 + 5 + 5)
        (0, (0, 5, 10, 30, 50, 100, 200, 400, 600), 1),
        (1, (0, 2, 4, 6, 8, 10, 12, 15, 20, 30, 40, 50, 70, 90, 120, 150, 200), 10),
        (2, (0, 2, 5, 10, 15, 20, 25, 30, 40, 50, 60, 80, 100, 130, 160, 200,
             250, 300, 350, 400), 10),
        (3, (0, 50, 100, 200, 400, 600), 1),
        (12, (0, 100, 200, 400, 1000, 2000), 1),
    ],
    211: [  # charged pion (533 voxels: 8 + 10*10 + 10*10 + 5 + 15*10 + 16*10 + 10)
        (0, (0, 5, 10, 30, 50, 100, 200, 400, 600), 1),
        (1, (0, 1, 4, 7, 10, 15, 30, 50, 90, 150, 200), 10),
        (2, (0, 5, 10, 20, 30, 50, 80, 130, 200, 300, 400), 10),
        (3, (0, 50, 100, 200, 400, 600), 1),
        (12, (0, 10, 20, 30, 50, 80, 100, 130, 160, 200, 250, 300, 350, 400,
              1000, 2000), 10),
        (13, (0, 10, 20, 30, 50, 80, 100, 130, 160, 200, 250, 300, 350, 400,
              600, 1000, 2000), 10),
        (14, (0, 50, 100, 150, 200, 250, 300, 400, 600, 1000, 2000), 1),
    ],
}
_ATLAS_CALO_CLASS_PID = (22, 211)          # class 0 = photon, class 1 = pion
_ATLAS_CALO_NCOLS = {pid: sum((len(e) - 1) * na for _, e, na in layers)
                     for pid, layers in _ATLAS_CALO_LAYERS.items()}   # {22:368, 211:533}
_ATLAS_CALO_NLAYERS = 24                    # rows: all ATLAS calo sampling layers
_ATLAS_CALO_NR = 24                         # columns: common radial bins
_ATLAS_CALO_R_EDGES = np.geomspace(1.0, 1600.0, _ATLAS_CALO_NR + 1)
_ATLAS_CALO_LOG_SCALE = 16.0                # ~log1p of the largest voxel energy
_ATLAS_CALO_TEST_FRACTION = 0.2
_ATLAS_CALO_SPLIT_SEED = 20180              # from the SIMU-2018-04 sample id
_ATLAS_CALO_REMOTE = (("photon_voxels.npy", "photon_samples.tgz", 22),
                      ("pion_voxels.npy", "pion_samples.tgz", 211))


def _atlas_calo_images(voxels, pid):
    """(n, ncols) float32 voxel energies -> (n, 1, 24, 24) float32 shower
    images in [0, 1]: sum over alpha, regrid radius onto the common axis,
    clip negatives, log1p-compress. Pure function — the unit tests exercise
    it directly on fabricated voxels."""
    voxels = np.asarray(voxels, dtype=np.float32)
    ncols = _ATLAS_CALO_NCOLS[pid]
    if voxels.ndim != 2 or voxels.shape[1] != ncols:
        raise ValueError(f"expected (n, {ncols}) voxels for pid {pid}, "
                         f"got {voxels.shape}")
    n = voxels.shape[0]
    img = np.zeros((n, _ATLAS_CALO_NLAYERS, _ATLAS_CALO_NR), dtype=np.float32)
    off = 0
    for lid, r_edges, na in _ATLAS_CALO_LAYERS[pid]:
        nr = len(r_edges) - 1
        # CSV is alpha-major within a layer (index = alpha * nr + r), verified
        # against the data; reshape to (n, alpha, r) and sum the azimuth away.
        radial = voxels[:, off:off + nr * na].reshape(n, na, nr).sum(axis=1)
        centers = 0.5 * (np.asarray(r_edges[:-1]) + np.asarray(r_edges[1:]))
        cols = np.clip(np.digitize(centers, _ATLAS_CALO_R_EDGES) - 1,
                       0, _ATLAS_CALO_NR - 1)
        for ri in range(nr):
            img[:, lid, cols[ri]] += radial[:, ri]
        off += nr * na
    np.clip(img, 0.0, None, out=img)
    np.log1p(img, out=img)
    img /= _ATLAS_CALO_LOG_SCALE
    np.clip(img, 0.0, 1.0, out=img)
    return np.ascontiguousarray(
        img.reshape(n, 1, _ATLAS_CALO_NLAYERS, _ATLAS_CALO_NR))


def _atlas_calo_split(n, class_index):
    """Deterministic per-class (train_rows, test_rows) index arrays into the
    class .npy — a fixed-seed shuffle then an 80/20 cut. Reproducible across
    processes and independent of any subset() seed."""
    rng = np.random.default_rng(_ATLAS_CALO_SPLIT_SEED + class_index)
    perm = rng.permutation(n)
    n_test = int(round(n * _ATLAS_CALO_TEST_FRACTION))
    return perm[n_test:], perm[:n_test]


def _atlas_calo_mmaps():
    d = data_dir() / "atlas_calo"
    paths = [d / f for f in DATASETS["atlas_calo"].files]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            "dataset 'atlas_calo' not downloaded — run: "
            + download_command("atlas_calo"))
    return [np.load(p, mmap_mode="r") for p in paths]


def _atlas_calo_load(subset_ns=None, seed=0):
    """load()/subset() core for atlas_calo. subset_ns=None -> the full split;
    (n_train, n_test) -> seeded stratified subsets. Builds images only for the
    rows it returns (memory-mapped voxel reads)."""
    mm = _atlas_calo_mmaps()
    splits = [_atlas_calo_split(mm[c].shape[0], c) for c in range(2)]

    def gather(which, take):
        pools = [splits[c][which] for c in range(2)]
        if take is None:
            sel = pools
        else:
            base, extra = divmod(take, 2)      # class 0 gets the odd sample
            counts = [base + (1 if c < extra else 0) for c in range(2)]
            rng = np.random.default_rng(seed + which)
            sel = []
            for c in range(2):
                if counts[c] > len(pools[c]):
                    raise ValueError(f"class {c} has only {len(pools[c])} "
                                     f"samples, need {counts[c]}")
                sel.append(rng.permutation(pools[c])[:counts[c]])
        Xs, ys = [], []
        for c in range(2):
            rows = np.sort(np.asarray(sel[c]))           # sorted mmap read
            vox = np.asarray(mm[c][rows])
            Xs.append(_atlas_calo_images(vox, _ATLAS_CALO_CLASS_PID[c]))
            ys.append(np.full(len(rows), c, dtype=np.int32))
        X = np.concatenate(Xs)
        y = np.concatenate(ys)
        # Deterministic shuffle so the two class blocks are interleaved.
        p = np.random.default_rng(1000 * (seed + 1) + which).permutation(len(y))
        return np.ascontiguousarray(X[p]), y[p]

    if subset_ns is None:
        Xtr, ytr = gather(0, None)
        Xte, yte = gather(1, None)
    else:
        Xtr, ytr = gather(0, subset_ns[0])
        Xte, yte = gather(1, subset_ns[1])
    return Xtr, ytr, Xte, yte


def _download_atlas_calo(d) -> None:
    """Fetch the record's photon/pion .tgz samples, verify each (announced
    length, gzip magic), parse the CSV voxel rows, and write a compact
    float32 voxel cache atomically. The raw .tgz is not kept — the cache is
    what load() reads."""
    d.mkdir(parents=True, exist_ok=True)
    base = DATASETS["atlas_calo"].base_url
    for out_name, tgz_name, pid in _ATLAS_CALO_REMOTE:
        out = d / out_name
        if out.is_file():
            print(f"atlas_calo: {out_name} already present")
            continue
        url = base + tgz_name
        print(f"atlas_calo: {url}\n  -> {out} (parsing voxels)")
        with urllib.request.urlopen(url, timeout=180) as r:
            length = r.headers.get("Content-Length")
            body = r.read()
        if length is not None and len(body) != int(length):
            raise OSError(f"{url}: truncated — got {len(body):,} of "
                          f"{int(length):,} announced bytes")
        if not body.startswith(b"\x1f\x8b"):
            raise OSError(f"{url}: not gzip data (starts {body[:4]!r}) — "
                          f"an error page or proxy response, not the dataset")
        ncols = _ATLAS_CALO_NCOLS[pid]
        rows = []
        with tarfile.open(fileobj=io.BytesIO(body), mode="r:gz") as tar:
            members = [m for m in tar.getmembers() if m.name.endswith(".csv")]
            if not members:
                raise OSError(f"{url}: no CSV members in archive")
            for m in sorted(members, key=lambda m: m.name):
                arr = np.loadtxt(tar.extractfile(m), delimiter=",",
                                 dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr[None, :]
                if arr.shape[1] != ncols:
                    raise OSError(f"{url}:{m.name}: expected {ncols} columns, "
                                  f"got {arr.shape[1]}")
                rows.append(arr)
        vox = np.concatenate(rows)
        tmp = d / ("." + out_name + ".part.npy")
        np.save(tmp, vox)
        tmp.replace(out)
        print(f"  done ({vox.shape[0]:,} events, {vox.nbytes:,} bytes cached)")


# -- public API ---------------------------------------------------------------

def load(name: str):
    """Load dataset ``name`` -> (X_train, y_train, X_test, y_test) as NCHW
    float32 in [0, 1], int32 labels.

    Never downloads: raises FileNotFoundError with the exact fix command if
    any file is missing. For a benchmark-sized slice prefer subset(), which
    converts only the slice to float32 (4x lower peak RAM on the full set).
    """
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    if name == "atlas_calo":
        return _atlas_calo_load()
    spec = DATASETS[name]
    d = data_dir() / name
    paths = [d / f for f in spec.files]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            f"dataset {name!r} not downloaded — run: {download_command(name)}")
    if name == "cifar10":
        Xtr, ytr, Xte, yte = _load_cifar10(paths[0])
    else:
        Xtr, ytr = _load_idx_pair(paths[0], paths[1])
        Xte, yte = _load_idx_pair(paths[2], paths[3])
    return _to_f01(Xtr), ytr, _to_f01(Xte), yte


def _load_u8(name: str):
    """load() without the float conversion: uint8 NCHW, int32 labels.

    Internal. subset() slices these and converts only the slice, so peak RAM
    is the raw bytes plus the subset — not a float32 copy of the full set
    (4x the bytes; measured 1.26 GB peak RSS on the cifar10 benchmark worker
    before this split, dataset load dominating every contender's number)."""
    if name not in DATASETS:
        raise KeyError(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}")
    spec = DATASETS[name]
    d = data_dir() / name
    paths = [d / f for f in spec.files]
    if not all(p.is_file() for p in paths):
        raise FileNotFoundError(
            f"dataset {name!r} not downloaded — run: {download_command(name)}")
    if name == "cifar10":
        return _load_cifar10(paths[0])
    Xtr, ytr = _load_idx_pair(paths[0], paths[1])
    Xte, yte = _load_idx_pair(paths[2], paths[3])
    return Xtr, ytr, Xte, yte


def subset(name: str, n_train: int, n_test: int, seed: int = 0):
    """Seeded stratified subset -> (X_train, y_train, X_test, y_test).

    Per-class quotas are as equal as the class counts allow (largest-
    remainder split of n over the classes present). The benchmark protocol
    uses subset("...", 2000, 1000, seed=0). Slices the raw uint8 arrays and
    converts only the slice to float32 — same values as slicing load()'s
    output, at a fraction of the peak RAM (see _load_u8).

    atlas_calo has no uint8 form; its subset reads only the selected voxel
    rows (memory-mapped) and builds shower images for just those."""
    if name == "atlas_calo":
        return _atlas_calo_load((n_train, n_test), seed)
    Xtr, ytr, Xte, yte = _load_u8(name)
    itr = _stratified_indices(ytr, n_train, np.random.default_rng(seed))
    ite = _stratified_indices(yte, n_test, np.random.default_rng(seed + 1))
    return (_to_f01(Xtr[itr]), ytr[itr],
            _to_f01(Xte[ite]), yte[ite])


def _stratified_indices(y, n, rng):
    classes = np.unique(y)
    base, extra = divmod(n, len(classes))
    picks = []
    for i, c in enumerate(classes):
        idx = np.flatnonzero(y == c)
        take = base + (1 if i < extra else 0)
        if take > len(idx):
            raise ValueError(f"class {c} has only {len(idx)} samples, need {take}")
        picks.append(rng.permutation(idx)[:take])
    return rng.permutation(np.concatenate(picks))


# -- explicit downloader (the only networking code) ----------------------------

def download(name: str) -> None:
    """Fetch every file of dataset ``name``, verified and atomic.

    The payload is checked before it can ever reach the load path: length
    against the server's Content-Length and the gzip magic (every dataset
    file, including CIFAR's .tar.gz, is gzip) — a truncated body or an HTML
    error page raises OSError instead of landing on disk. The verified body
    is written to a ``.part`` file and renamed into place, so ``load()``
    never sees a partial download.
    """
    d = data_dir() / name
    if name == "atlas_calo":
        _download_atlas_calo(d)
        return
    spec = DATASETS[name]
    d.mkdir(parents=True, exist_ok=True)
    for fname in spec.files:
        path = d / fname
        if path.is_file():
            print(f"{name}: {fname} already present")
            continue
        url = spec.base_url + fname
        print(f"{name}: {url}\n  -> {path}")
        with urllib.request.urlopen(url, timeout=60) as r:
            length = r.headers.get("Content-Length")
            body = r.read()
        if length is not None and len(body) != int(length):
            raise OSError(f"{url}: truncated — got {len(body):,} of "
                          f"{int(length):,} announced bytes")
        if not body.startswith(b"\x1f\x8b"):
            raise OSError(f"{url}: not gzip data (starts {body[:4]!r}) — "
                          f"an error page or proxy response, not the dataset")
        tmp = path.with_name(fname + ".part")
        tmp.write_bytes(body)
        tmp.replace(path)
        print(f"  done ({len(body):,} bytes)")


def _main(argv) -> int:
    if len(argv) == 1 and argv[0] == "list":
        for name, spec in DATASETS.items():
            d = data_dir() / name
            state = "present" if all((d / f).is_file() for f in spec.files) else "missing"
            print(f"{name:14} {state:8} {spec.note}")
        return 0
    if len(argv) == 2 and argv[0] == "download":
        names = list(DATASETS) if argv[1] == "all" else [argv[1]]
        failed = []
        for name in names:
            if name not in DATASETS:
                print(f"unknown dataset {name!r}; available: {', '.join(DATASETS)}",
                      file=sys.stderr)
                return 2
            try:
                download(name)
            except Exception as exc:            # keep fetching the rest
                print(f"{name}: FAILED — {exc}", file=sys.stderr)
                failed.append(name)
        if failed:
            print(f"download failed for: {', '.join(failed)}", file=sys.stderr)
            return 1
        return 0
    print("usage: python -m mantissa_cnn.datasets download <name|all>\n"
          "       python -m mantissa_cnn.datasets list", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))
