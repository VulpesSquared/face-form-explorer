import numpy as np
import pandas as pd

from face_analysis.similarity import build_similarity_matrix


def test_single_person_self_similarity_is_100():
    people = pd.DataFrame({"person": ["User label"], "face_width_height": [0.7]})
    assert build_similarity_matrix(people).iloc[0, 0] == 100.0


def test_similarity_is_finite_symmetric_and_bounded():
    people = pd.DataFrame({"person": ["A", "B", "C"], "face_width_height": [0.7, 0.8, 0.9]})
    matrix = build_similarity_matrix(people).to_numpy()
    np.testing.assert_allclose(matrix, matrix.T)
    np.testing.assert_allclose(matrix.diagonal(), 100)
    assert np.isfinite(matrix).all()
    assert ((matrix >= 0) & (matrix <= 100)).all()
