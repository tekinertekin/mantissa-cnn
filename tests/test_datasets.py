"""Dataset loader tests on tiny fabricated IDX/CIFAR files — no network,
no real downloads. The 'not downloaded' error message is asserted verbatim."""
import gzip
import io
import pickle
import struct
import tarfile

import numpy as np
import pytest

from mantissa_cnn import datasets


# -- fabrication helpers --------------------------------------------------------

def _gz(path, payload: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wb") as f:
        f.write(payload)


def _idx3_images(arr):                       # arr: uint8 (n, 28, 28)
    return struct.pack(">IIII", 0x00000803, *arr.shape) + arr.tobytes()


def _idx1_labels(y):                         # y: uint8 (n,)
    return struct.pack(">II", 0x00000801, len(y)) + y.tobytes()


def _idx2_int_labels(y):                     # QMNIST: int32 (n, 8), class = col 0
    full = np.zeros((len(y), 8), dtype=">i4")
    full[:, 0] = y
    return struct.pack(">III", 0x00000C02, len(y), 8) + full.tobytes()


def _fake_mnist_family(root, name, n_train=40, n_test=20, labeler=_idx1_labels,
                       files=datasets._IDX4):
    rng = np.random.default_rng(0)
    d = root / name
    for fname, n in ((files[0], n_train), (files[2], n_test)):
        _gz(d / fname, _idx3_images(
            rng.integers(0, 256, size=(n, 28, 28), dtype=np.uint8)))
    for fname, n in ((files[1], n_train), (files[3], n_test)):
        _gz(d / fname, labeler((np.arange(n) % 10).astype(np.uint8)))


def _fake_cifar10(root, n_per_batch=20):
    rng = np.random.default_rng(1)
    d = root / "cifar10"
    d.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for member in [f"data_batch_{i}" for i in range(1, 6)] + ["test_batch"]:
            body = pickle.dumps({
                b"data": rng.integers(0, 256, size=(n_per_batch, 3072),
                                      dtype=np.uint8),
                b"labels": (np.arange(n_per_batch) % 10).tolist()})
            info = tarfile.TarInfo(f"cifar-10-batches-py/{member}")
            info.size = len(body)
            tar.addfile(info, io.BytesIO(body))
    (d / "cifar-10-python.tar.gz").write_bytes(buf.getvalue())


def _fake_atlas_calo(root, n_photon=80, n_pion=80):
    """Tiny fabricated voxel caches: (n, 368) photon and (n, 533) pion float32
    energies, the widths the real record's binning.xml implies."""
    rng = np.random.default_rng(3)
    d = root / "atlas_calo"
    d.mkdir(parents=True, exist_ok=True)
    for name, n, pid in (("photon_voxels.npy", n_photon, 22),
                         ("pion_voxels.npy", n_pion, 211)):
        ncols = datasets._ATLAS_CALO_NCOLS[pid]
        np.save(d / name, (rng.random((n, ncols), dtype=np.float32) * 1000.0))


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MANTISSA_CNN_DATA", str(tmp_path))
    return tmp_path


# -- tests -----------------------------------------------------------------------

def test_registry_names():
    assert set(datasets.DATASETS) == {
        "mnist", "fashion_mnist", "kmnist", "cifar10", "qmnist", "atlas_calo"}


def test_not_downloaded_message_verbatim(data_root):
    with pytest.raises(FileNotFoundError) as exc:
        datasets.load("mnist")
    assert str(exc.value) == ("dataset 'mnist' not downloaded — run: "
                              "python -m mantissa_cnn.datasets download mnist")


def test_unknown_dataset_raises_keyerror(data_root):
    with pytest.raises(KeyError, match="unknown dataset"):
        datasets.load("imagenet")


@pytest.mark.parametrize("name", ["mnist", "fashion_mnist", "kmnist"])
def test_idx_family_loads_nchw_float32(data_root, name):
    _fake_mnist_family(data_root, name)
    Xtr, ytr, Xte, yte = datasets.load(name)
    assert Xtr.shape == (40, 1, 28, 28) and Xte.shape == (20, 1, 28, 28)
    assert Xtr.dtype == np.float32 and Xtr.flags["C_CONTIGUOUS"]
    assert 0.0 <= Xtr.min() and Xtr.max() <= 1.0
    assert ytr.dtype == np.int32 and yte.dtype == np.int32
    assert set(np.unique(ytr)) <= set(range(10))


def test_qmnist_idx2_int_labels(data_root):
    _fake_mnist_family(data_root, "qmnist", labeler=_idx2_int_labels,
                       files=datasets.DATASETS["qmnist"].files)
    Xtr, ytr, Xte, yte = datasets.load("qmnist")
    assert Xtr.shape == (40, 1, 28, 28)
    assert ytr.dtype == np.int32
    assert np.array_equal(ytr, np.arange(40) % 10)   # column 0 of the idx2 table


def test_cifar10_loads_from_tarball(data_root):
    _fake_cifar10(data_root)
    Xtr, ytr, Xte, yte = datasets.load("cifar10")
    assert Xtr.shape == (100, 3, 32, 32) and Xte.shape == (20, 3, 32, 32)
    assert Xtr.dtype == np.float32 and 0.0 <= Xtr.min() and Xtr.max() <= 1.0
    assert ytr.dtype == np.int32 and len(ytr) == 100


def test_idx_magic_number_is_verified(data_root):
    d = data_root / "mnist"
    _fake_mnist_family(data_root, "mnist")
    _gz(d / "train-images-idx3-ubyte.gz", b"\xff\xff\xff\xff garbage")
    with pytest.raises(ValueError, match="bad IDX magic"):
        datasets.load("mnist")


def test_idx_size_is_verified(data_root):
    _fake_mnist_family(data_root, "mnist")
    d = data_root / "mnist"
    # header claims 99 images but carries 40
    rng = np.random.default_rng(2)
    body = struct.pack(">IIII", 0x00000803, 99, 28, 28) + \
        rng.integers(0, 256, size=(40, 28, 28), dtype=np.uint8).tobytes()
    _gz(d / "train-images-idx3-ubyte.gz", body)
    with pytest.raises(ValueError, match="size"):
        datasets.load("mnist")


def test_subset_is_stratified_and_seeded(data_root):
    _fake_mnist_family(data_root, "mnist", n_train=100, n_test=60)
    Xtr, ytr, Xte, yte = datasets.subset("mnist", 20, 10, seed=0)
    assert Xtr.shape == (20, 1, 28, 28) and Xte.shape == (10, 1, 28, 28)
    counts = np.bincount(ytr, minlength=10)
    assert counts.tolist() == [2] * 10                 # exactly stratified
    assert np.bincount(yte, minlength=10).tolist() == [1] * 10
    Xtr2, ytr2, _, _ = datasets.subset("mnist", 20, 10, seed=0)
    assert np.array_equal(Xtr, Xtr2) and np.array_equal(ytr, ytr2)
    _, ytr3, _, _ = datasets.subset("mnist", 20, 10, seed=5)
    assert not np.array_equal(ytr, ytr3)               # seed matters


def test_download_command_matches_error_message():
    assert datasets.download_command("cifar10") == \
        "python -m mantissa_cnn.datasets download cifar10"


# -- atlas_calo -----------------------------------------------------------------

def test_atlas_calo_not_downloaded_message_verbatim(data_root):
    with pytest.raises(FileNotFoundError) as exc:
        datasets.load("atlas_calo")
    assert str(exc.value) == ("dataset 'atlas_calo' not downloaded — run: "
                              "python -m mantissa_cnn.datasets download atlas_calo")


def test_atlas_calo_loads_nchw_float32_two_class(data_root):
    _fake_atlas_calo(data_root, n_photon=80, n_pion=80)
    Xtr, ytr, Xte, yte = datasets.load("atlas_calo")
    # 20% test per class: 16 test + 64 train per class, both classes.
    assert Xtr.shape == (128, 1, 24, 24) and Xte.shape == (32, 1, 24, 24)
    assert Xtr.dtype == np.float32 and Xtr.flags["C_CONTIGUOUS"]
    assert 0.0 <= Xtr.min() and Xtr.max() <= 1.0
    assert ytr.dtype == np.int32 and yte.dtype == np.int32
    assert set(np.unique(ytr)) == {0, 1}
    assert np.bincount(ytr).tolist() == [64, 64]      # train/test disjoint,
    assert np.bincount(yte).tolist() == [16, 16]      # both stratified


def test_atlas_calo_subset_is_stratified_and_seeded(data_root):
    _fake_atlas_calo(data_root, n_photon=120, n_pion=120)
    Xtr, ytr, Xte, yte = datasets.subset("atlas_calo", 40, 20, seed=0)
    assert Xtr.shape == (40, 1, 24, 24) and Xte.shape == (20, 1, 24, 24)
    assert np.bincount(ytr, minlength=2).tolist() == [20, 20]
    assert np.bincount(yte, minlength=2).tolist() == [10, 10]
    Xtr2, ytr2, _, _ = datasets.subset("atlas_calo", 40, 20, seed=0)
    assert np.array_equal(Xtr, Xtr2) and np.array_equal(ytr, ytr2)
    _, ytr3, _, _ = datasets.subset("atlas_calo", 40, 20, seed=5)
    assert not np.array_equal(ytr, ytr3)              # seed matters


def test_atlas_calo_subset_train_test_are_disjoint(data_root):
    # Same seed for train and test pools must still draw from disjoint rows:
    # a photon shower placed in train can never reappear in test.
    _fake_atlas_calo(data_root, n_photon=100, n_pion=100)
    Xtr, ytr, Xte, yte = datasets.subset("atlas_calo", 40, 20, seed=0)
    tr = {r.tobytes() for r in Xtr}
    assert not (tr & {r.tobytes() for r in Xte})


def test_atlas_calo_image_build_rejects_wrong_width():
    with pytest.raises(ValueError, match=r"expected \(n, 368\) voxels for pid 22"):
        datasets._atlas_calo_images(np.zeros((2, 100), np.float32), 22)


def test_atlas_calo_alpha_sum_and_radial_regrid():
    # One photon event with unit energy in a single (layer 2, alpha-major)
    # voxel lands in that layer's row, at the radial column its bin maps to,
    # and nowhere else — verifies the alpha-major reshape and radial digitize.
    pid = 22
    ncols = datasets._ATLAS_CALO_NCOLS[pid]
    vox = np.zeros((1, ncols), np.float32)
    # layer offsets: 8 (L0) + 160 (L1) then L2 begins; alpha=0, r-bin=0.
    vox[0, 8 + 160] = np.expm1(16.0)                  # log1p/scale -> ~1.0
    img = datasets._atlas_calo_images(vox, pid)[0, 0]
    assert img.shape == (24, 24)
    hot = np.argwhere(img > 1e-6)
    assert hot.shape == (1, 1 * 2) and hot[0, 0] == 2  # exactly one cell, row=layer 2
    assert abs(float(img[2, hot[0, 1]]) - 1.0) < 1e-3
