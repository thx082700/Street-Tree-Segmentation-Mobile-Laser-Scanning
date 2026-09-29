import zipfile
from pathlib import Path

import numpy as np
import pytest

from sparsetree3d.io import (
    build_features,
    competition_stem,
    make_submission_archive,
    save_prediction,
    validate_prediction_file,
)


def test_build_features_is_finite() -> None:
    xyz = np.array([[1, 2, 3], [1, 2, 4], [2, 2, 5]], dtype=np.float32)
    intensity = np.array([10, 20, 30], dtype=np.float32)
    features = build_features(xyz, intensity)
    assert features.shape == (3, 4)
    assert np.isfinite(features).all()
    assert np.all(features[:, 0] == 1)


def test_competition_stem() -> None:
    assert competition_stem(Path("dataset/05/PCD/3.ply")) == "05_3"
    assert competition_stem(Path("05_3.ply")) == "05_3"


def test_submission_archive(tmp_path: Path) -> None:
    first = save_prediction(tmp_path / "05_3.npy", np.array([-1, 1, 1]))
    second = save_prediction(tmp_path / "13_2.npy", np.array([[0], [2], [2]]))
    assert validate_prediction_file(first) == (3, 1)
    saved = np.load(first, allow_pickle=False)
    assert saved.shape == (3, 1)
    assert saved.dtype == np.int16
    archive = make_submission_archive([second, first], tmp_path / "submission.zip")
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.namelist() == ["05_3.npy", "13_2.npy"]


def test_rejects_invalid_prediction_shape(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_prediction(tmp_path / "05_3.npy", np.zeros((2, 2)))
