from dataclasses import FrozenInstanceError

import pytest

from birsvd.svd.parameter import BIRSVDParameter, DEFAULT_PARAM


def test_birsvd_parameter_defaults():
    param = BIRSVDParameter()

    assert param.n_iter == 30
    assert param.init_method == "randOrthoNormal"
    assert param.r_type_L == "2ndOrderDiff_acc8"
    assert param.r_degree_L == 0.0001
    assert param.r_type_R == "2ndOrderDiff_acc8"
    assert param.r_degree_R == 0.0001
    assert param.lsqr_niter == 25


def test_default_param_matches_default_instance():
    assert DEFAULT_PARAM == BIRSVDParameter()


@pytest.mark.parametrize(
    "init_method",
    [
        "zeroOneVectors",
        "randOrthoNormal",
        "polyOrthoNormal",
    ],
)
def test_birsvd_parameter_accepts_valid_init_methods(init_method):
    param = BIRSVDParameter(init_method=init_method)

    assert param.init_method == init_method


@pytest.mark.parametrize(
    "r_type",
    [
        "TiKh",
        "2ndOrderDiff_acc2",
        "2ndOrderDiff_acc4",
        "2ndOrderDiff_acc6",
        "2ndOrderDiff_acc8",
    ],
)
def test_birsvd_parameter_accepts_valid_regularization_types(r_type):
    param = BIRSVDParameter(r_type_L=r_type, r_type_R=r_type)

    assert param.r_type_L == r_type
    assert param.r_type_R == r_type


def test_birsvd_parameter_accepts_custom_values():
    param = BIRSVDParameter(
        n_iter=10,
        init_method="polyOrthoNormal",
        r_type_L="TiKh",
        r_degree_L=0.2,
        r_type_R="2ndOrderDiff_acc4",
        r_degree_R=0.3,
        lsqr_niter=5,
    )

    assert param.n_iter == 10
    assert param.init_method == "polyOrthoNormal"
    assert param.r_type_L == "TiKh"
    assert param.r_degree_L == 0.2
    assert param.r_type_R == "2ndOrderDiff_acc4"
    assert param.r_degree_R == 0.3
    assert param.lsqr_niter == 5


def test_birsvd_parameter_is_frozen():
    param = BIRSVDParameter()

    with pytest.raises(FrozenInstanceError):
        param.n_iter = 10


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"init_method": "invalid"}, "Invalid init_method"),
        ({"r_type_L": "invalid"}, "Invalid r_type_L"),
        ({"r_type_R": "invalid"}, "Invalid r_type_R"),
    ],
)
def test_birsvd_parameter_rejects_invalid_options(kwargs, message):
    with pytest.raises(ValueError, match=message):
        BIRSVDParameter(**kwargs)
