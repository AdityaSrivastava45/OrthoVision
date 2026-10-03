"""Ordering + geometry tests for the Stage-2 preprocessing package."""

import numpy as np
import pytest

from orthovision.preprocessing.geometry import (
    ORDERING_INSTANCE,
    ORDERING_SPATIAL,
    SliceRow,
    order_slices,
)

IOP_SAG = (0.0, 1.0, 0.0, 0.0, 0.0, -1.0)
IOP_COR = (1.0, 0.0, 0.0, 0.0, 0.0, -1.0)
IOP_AX = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)


def mk(sop, inst, x, extra=None):
    return SliceRow(
        sop_uid=sop,
        path=f"{sop}.dcm",
        instance_number=inst,
        iop=IOP_SAG,
        ipp=(x, 0.0, 0.0),
        rows=64,
        columns=64,
        pixel_spacing=(0.4, 0.4),
        rescale_slope=None,
        photometric="MONOCHROME2",
        extra=extra or {},
    )


def test_spatial_order_ascending_position():
    slices = [mk(f"s{i}", i + 1, x=float(10 - 2 * i)) for i in range(5)]
    ordered, geo = order_slices(slices)
    assert geo.ordering_method == ORDERING_SPATIAL
    normal = np.asarray(geo.normal)
    projections = [float(np.dot(np.asarray(s.ipp), normal)) for s in ordered]
    assert projections == sorted(projections)
    assert geo.positions_mm == pytest.approx(projections)


def test_ordering_is_deterministic_under_input_permutation():
    slices = [mk(f"s{i}", (i * 7 + 3) % 5, x=float(i * 3)) for i in range(5)]
    runs = [tuple(order_slices(list(permutation))[1].positions_mm) for permutation in
            (slices, list(reversed(slices)), [slices[2], slices[0], slices[4], slices[1], slices[3]])]
    assert runs[0] == runs[1] == runs[2]


def test_positions_aligned_with_returned_order_not_input_order():
    slices = [mk("a", 5, 8.0), mk("b", 4, 6.0), mk("c", 3, 4.0), mk("d", 2, 2.0), mk("e", 1, 0.0)]
    ordered, geo = order_slices(slices)
    normal = np.asarray(geo.normal)
    projections = {s.sop_uid: float(np.dot(np.asarray(s.ipp), normal)) for s in slices}
    returned = [projections[s.sop_uid] for s in ordered]
    assert returned == sorted(returned)
    assert geo.positions_mm == pytest.approx(returned)
    assert len({s.sop_uid for s in ordered}) == 5


def test_fallback_instance_number_when_geometry_missing():
    rows = [
        SliceRow("x1", "p1", 2, None, None, 32, 32, None, None, "MONOCHROME2"),
        SliceRow("x2", "p2", 1, None, None, 32, 32, None, None, "MONOCHROME2"),
    ]
    ordered, geo = order_slices(rows)
    assert geo.ordering_method == ORDERING_INSTANCE
    assert [s.instance_number for s in ordered] == [1, 2]
    assert geo.normal is None


def test_laterality_majority_ignores_nan_and_empty():
    slices = [
        mk("a", 1, 0.0, {"laterality": "L"}),
        mk("b", 2, 1.0, {"laterality": float("nan")}),
        mk("c", 3, 2.0, {"laterality": ""}),
        mk("d", 4, 3.0, {"laterality": "l"}),
    ]
    _, geo = order_slices(slices)
    assert geo.laterality == "L"


def test_duplicate_positions_counted():
    slices = [mk("a", 1, 0.0), mk("b", 2, 0.00001), mk("c", 3, 5.0)]
    _, geo = order_slices(slices)
    assert geo.duplicate_positions == 1


@pytest.mark.parametrize("iop,axis", [(IOP_SAG, 0), (IOP_COR, 1), (IOP_AX, 2)])
def test_normal_axis_per_plane(iop, axis):
    row = SliceRow("s", "p", 1, iop, (0.0, 0.0, 0.0), 16, 16, (0.5, 0.5), None, "MONOCHROME2")
    _, geo = order_slices([row])
    normal = np.asarray(geo.normal)
    assert abs(normal[axis]) > 0.99
