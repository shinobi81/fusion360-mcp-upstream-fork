"""Sketch curves must land in the sketch plane, whatever ``z`` the caller sends.

``SketchCurves`` take points in the sketch's **own** coordinate system, so a
caller-supplied ``z`` composes with the plane's transform. On a sketch offset to
model ``z=-0.65`` (``create_sketch(z_offset=-0.65)``), drawing with
``center_z=-0.65`` used to put the circle at model ``z=-1.30`` — the offset
applied twice, reported as success.

The add-in cannot be imported under pytest because it needs Fusion's ``adsk``
runtime, so these tests load ``command_handler.py`` against a minimal fake and
assert on what is handed to the sketch curves — the geometry Fusion actually
builds.
"""

from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
SERVER_DIR = REPO_ROOT / "addon" / "server"
FAKE_PACKAGE = "addon_under_test"


# ----------------------------------------------------------------------
# Fake adsk / sketch recording surface
# ----------------------------------------------------------------------


class FakePoint3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    @classmethod
    def create(cls, x=0.0, y=0.0, z=0.0):
        return cls(x, y, z)

    def coords(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"FakePoint3D({self.x}, {self.y}, {self.z})"


class FakeVector3D:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = float(x), float(y), float(z)

    @classmethod
    def create(cls, x=0.0, y=0.0, z=0.0):
        return cls(x, y, z)


class FakeTransform:
    """A sketch transform, offset along Z and optionally rotated.

    ``axes`` lets a test stand in for a sketch on the yz/xz plane, where the
    sketch's own x/y are not model x/y — the one case where transposing the
    axis math in ``_model_point`` would go unnoticed.
    """

    def __init__(self, offset=0.0, x_axis=(1.0, 0.0, 0.0), y_axis=(0.0, 1.0, 0.0)):
        self.offset = float(offset)
        self.x_axis = x_axis
        self.y_axis = y_axis

    def getAsCoordinateSystem(self):
        return (
            FakePoint3D(0.0, 0.0, self.offset),
            FakeVector3D(*self.x_axis),
            FakeVector3D(*self.y_axis),
            FakeVector3D(0.0, 0.0, 1.0),
        )


class _Recorder:
    """Stands in for one sketch-curve collection, recording calls."""

    def __init__(self, calls, kind):
        self._calls, self._kind = calls, kind

    def addByCenterRadius(self, center, radius):
        self._calls.append((self._kind, {"center": center, "radius": radius}))

    def addTwoPointRectangle(self, p1, p2):
        self._calls.append((self._kind, {"points": [p1, p2]}))

    def addByTwoPoints(self, p1, p2):
        self._calls.append((self._kind, {"points": [p1, p2]}))

    def addByCenterStartSweep(self, center, start, sweep):
        self._calls.append(
            (self._kind, {"center": center, "start": start, "sweep": sweep})
        )

    def add(self, points, *args):
        self._calls.append((self._kind, {"points": list(points)}))


class FakeSketchCurves:
    def __init__(self, calls):
        self.sketchCircles = _Recorder(calls, "circle")
        self.sketchLines = _Recorder(calls, "line")
        self.sketchArcs = _Recorder(calls, "arc")
        self.sketchFittedSplines = _Recorder(calls, "fit_spline")
        self.sketchControlPointSplines = _Recorder(calls, "control_spline")


class FakeSketch:
    def __init__(self, offset=0.0, name="Sketch1", transform=None):
        self.name = name
        self.isComputeDeferred = False
        self.transform = transform if transform is not None else FakeTransform(offset)
        self.calls: list[tuple[str, dict]] = []
        self.sketchCurves = FakeSketchCurves(self.calls)

    def last_call(self, kind=None):
        """The single (kind, payload) recorded, asserting one call happened."""
        assert len(self.calls) == 1, f"expected one curve call, got {self.calls}"
        kind_recorded, payload = self.calls[0]
        if kind is not None:
            assert kind_recorded == kind, f"expected a {kind}, got {kind_recorded}"
        return payload


class FakeObjectCollection:
    def __init__(self):
        self.items = []

    @classmethod
    def create(cls):
        return cls()

    def add(self, item):
        self.items.append(item)

    def __iter__(self):
        return iter(self.items)

    def __len__(self):
        return len(self.items)


# ----------------------------------------------------------------------
# Module loading with a fake adsk
# ----------------------------------------------------------------------


def _module(name: str, **attrs) -> types.ModuleType:
    """A stub module carrying *attrs* (setattr keeps type checkers quiet)."""
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


def _install_fake_adsk():
    core = _module(
        "adsk.core",
        Point3D=FakePoint3D,
        Vector3D=FakeVector3D,
        ObjectCollection=FakeObjectCollection,
    )
    fusion = _module("adsk.fusion")
    cam = _module("adsk.cam")
    adsk = _module("adsk", core=core, fusion=fusion, cam=cam)
    sys.modules.update(
        {
            "adsk": adsk,
            "adsk.core": core,
            "adsk.fusion": fusion,
            "adsk.cam": cam,
        }
    )


@pytest.fixture(scope="module")
def handler_module():
    """command_handler.py, imported against the fake runtime.

    The fake ``adsk`` is torn down afterwards: leaving it in ``sys.modules``
    would hand a fake to any later test that imports an adsk-dependent module
    instead of failing loudly.
    """
    if FAKE_PACKAGE in sys.modules:
        return sys.modules[f"{FAKE_PACKAGE}.server.command_handler"]

    _install_fake_adsk()
    package = types.ModuleType(FAKE_PACKAGE)
    package.__path__ = [str(SERVER_DIR.parent)]
    server = types.ModuleType(f"{FAKE_PACKAGE}.server")
    server.__path__ = [str(SERVER_DIR)]
    setattr(server, "get_logger", lambda name=None: logging.getLogger("addon-test"))
    sys.modules[FAKE_PACKAGE] = package
    sys.modules[f"{FAKE_PACKAGE}.server"] = server
    yield importlib.import_module(f"{FAKE_PACKAGE}.server.command_handler")
    for name in ("adsk", "adsk.core", "adsk.fusion", "adsk.cam"):
        sys.modules.pop(name, None)


@pytest.fixture
def draw(handler_module):
    """Return ``(sketch, handler)``: a fake sketch and a bound handler."""

    def factory(offset=-0.65, x_axis=(1.0, 0.0, 0.0), y_axis=(0.0, 1.0, 0.0)):
        handler = handler_module.CommandHandler.__new__(handler_module.CommandHandler)
        sketch = FakeSketch(
            offset=offset,
            transform=FakeTransform(offset=offset, x_axis=x_axis, y_axis=y_axis),
        )
        handler._last_sketch = lambda: sketch
        return sketch, handler

    return factory


# ----------------------------------------------------------------------
# The bug: a z that duplicates the plane offset was applied a second time
# ----------------------------------------------------------------------


class TestGeometryLandsOnTheSketchPlane:
    def test_draw_circle_drops_a_z_that_duplicates_the_offset(self, draw):
        sketch, handler = draw(offset=-0.65)

        result = handler.draw_circle(radius=0.3, center_x=5, center_y=5, center_z=-0.65)

        payload = sketch.last_call("circle")
        assert payload["center"].coords() == (5.0, 5.0, 0.0)
        assert payload["radius"] == 0.3
        # The circle is on the plane, so its model position is the plane's.
        assert result["model_center"][2] == pytest.approx(-0.65)
        assert result["ignored_z"] == -0.65
        assert "sketch plane" in result["note"]

    def test_model_position_follows_the_sketch_axes(self, draw):
        """On a yz-plane sketch, sketch x maps to model y and sketch y to model z.

        With identity axes everywhere else in these fakes, a transposed
        ``x_axis.y``/``y_axis.x`` in the model-position math would pass every
        other test here.
        """
        sketch, handler = draw(
            offset=0.0, x_axis=(0.0, 1.0, 0.0), y_axis=(0.0, 0.0, 1.0)
        )

        result = handler.draw_circle(radius=1, center_x=2, center_y=3)

        assert result["model_center"] == pytest.approx([0.0, 2.0, 3.0])

    def test_draw_rectangle_drops_a_z_that_duplicates_the_offset(self, draw):
        sketch, handler = draw(offset=-0.65)

        handler.draw_rectangle(width=2, height=2, origin_z=-0.65)

        p1, p2 = sketch.last_call("line")["points"]
        assert p1.coords() == (0.0, 0.0, 0.0)
        assert p2.coords() == (2.0, 2.0, 0.0)

    def test_draw_line_drops_both_z_values(self, draw):
        sketch, handler = draw(offset=-0.65)

        handler.draw_line(
            start_x=0, start_y=0, end_x=2, end_y=0, start_z=-0.65, end_z=-0.65
        )

        start, end = sketch.last_call("line")["points"]
        assert start.coords() == (0.0, 0.0, 0.0)
        assert end.coords() == (2.0, 0.0, 0.0)

    def test_draw_arc_drops_both_z_values(self, draw):
        sketch, handler = draw(offset=-0.65)

        handler.draw_arc(
            center_x=0,
            center_y=0,
            start_x=1,
            start_y=0,
            sweep_angle=90,
            center_z=-0.65,
            start_z=-0.65,
        )

        payload = sketch.last_call("arc")
        assert payload["center"].coords() == (0.0, 0.0, 0.0)
        assert payload["start"].coords() == (1.0, 0.0, 0.0)

    def test_draw_spline_drops_a_z_on_every_point(self, draw):
        sketch, handler = draw(offset=-0.65)

        handler.draw_spline(
            spline_type="fit_points", points=[[0, 0, -0.65], [1, 1, -0.65]]
        )

        points = sketch.last_call("fit_spline")["points"]
        assert [p.coords() for p in points] == [(0.0, 0.0, 0.0), (1.0, 1.0, 0.0)]

    def test_create_polygon_drops_a_z_that_duplicates_the_offset(self, draw):
        sketch, handler = draw(offset=-0.65)

        handler.create_polygon(sides=4, radius=1, center_z=-0.65)

        assert all(call[1]["points"][0].z == 0.0 for call in sketch.calls)
        assert len(sketch.calls) == 4


class TestDispatch:
    def test_execute_command_wraps_the_draw_in_a_success_envelope(self, draw):
        """The schema -> handler kwarg path and the envelope, end to end."""
        sketch, handler = draw(offset=-0.65)
        handler._snapshot = lambda: None  # no design behind this fake

        out = handler.execute_command(
            {"type": "draw_circle", "params": {"radius": 1, "center_z": -0.65}}
        )

        assert out["status"] == "success"
        assert out["result"]["ok"] is True
        assert out["result"]["ignored_z"] == -0.65
        assert sketch.last_call("circle")["center"].coords() == (0.0, 0.0, 0.0)


class TestNothingChangesWhenZIsOmitted:
    def test_default_draw_rectangle_reports_no_ignored_z(self, draw):
        sketch, handler = draw(offset=-0.65)

        result = handler.draw_rectangle(width=2, height=2)

        p1, p2 = sketch.last_call("line")["points"]
        assert p1.coords() == (0.0, 0.0, 0.0)
        assert p2.coords() == (2.0, 2.0, 0.0)
        assert "ignored_z" not in result

    def test_unoffset_sketch_model_position_matches_sketch_position(self, draw):
        sketch, handler = draw(offset=0.0)

        result = handler.draw_circle(radius=1, center_x=3, center_y=4)

        assert result["model_center"] == pytest.approx([3.0, 4.0, 0.0])
