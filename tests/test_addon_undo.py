"""``undo`` must do something it can verify, and say so honestly when it can't.

Measured against Fusion 2706 (`command_handler.undo` in 0.3.1):

* ``UndoCommand.execute()`` is **asynchronous**. Four of five consecutive calls
  returned ``undone: true`` while the design was untouched, and 57 event pumps
  inside one handler did not make a queued undo land — it applies only after
  the handler returns.
* The design-type guard therefore compared ``designType`` before the undo
  landed, i.e. it could never fire.

So undo is now two paths:

* **parametric** — move ``timeline.markerPosition`` back one step. The API is
  synchronous, so the change is observable and verifiable in the same call. The
  feature stays in the timeline (suppressed) and returns if the marker moves
  forward again.
* **direct / no timeline** — Fusion has no synchronous handle, so the UI command
  is issued and reported as ``undone: "issued"`` with ``verified: False`` rather
  than claiming success nobody checked.

The add-in cannot be imported under pytest (it needs Fusion's ``adsk``), so
these tests drive ``command_handler.py`` against a fake design.
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
FAKE_PACKAGE = "addon_undo_under_test"

PARAMETRIC = "ParametricDesignType"
DIRECT = "DirectDesignType"


# ----------------------------------------------------------------------
# Fake design
# ----------------------------------------------------------------------


class FakeTimeline:
    def __init__(self, design):
        self._design = design

    @property
    def count(self):
        return self._design.timeline_count

    @property
    def markerPosition(self):
        return self._design.marker

    @markerPosition.setter
    def markerPosition(self, value):
        self._design.set_marker(value)


class _Counted:
    def __init__(self, design, attr):
        self._design = design
        self._attr = attr

    @property
    def count(self):
        return getattr(self._design, self._attr)


class FakeRoot:
    def __init__(self, design):
        self.bRepBodies = _Counted(design, "bodies")
        self.sketches = _Counted(design, "sketches")
        self.features = _Counted(design, "features")


class FakeDesign:
    def __init__(
        self,
        features=2,
        marker=None,
        design_type=PARAMETRIC,
        marker_switches_type=False,
        timeline_raises=None,
    ):
        self.timeline_count = features
        self.marker = features if marker is None else marker
        self.bodies = 1 if features else 0
        self.sketches = 1 if features else 0
        self.features = features
        self.designType = design_type
        self.marker_switches_type = marker_switches_type
        # Direct designs raise on design.timeline in Fusion, so the fake does
        # too unless a test says otherwise: a more permissive fake hides bugs.
        self.timeline_raises = (
            design_type == DIRECT if timeline_raises is None else timeline_raises
        )
        self._timeline = FakeTimeline(self)
        self.rootComponent = FakeRoot(self)
        self.undo_commands_executed = 0
        self.marker_writes = []
        self.parentDocument = type("FakeDocument", (), {"name": "fake-design"})()

    @property
    def timeline(self):
        """Fusion raises here in direct mode: "this is not a parametric design"."""
        if self.timeline_raises:
            raise RuntimeError("3 : this is not a parametric design")
        return self._timeline

    def set_marker(self, value):
        self.marker_writes.append(value)
        self.marker = value
        if self.marker_switches_type:
            self.designType = DIRECT


class FakeCommandDefinition:
    def __init__(self, name, on_execute):
        self.name = name
        self._on_execute = on_execute

    def execute(self):
        self._on_execute()


class FakeCommandDefinitions:
    def __init__(self, defs):
        self._defs = defs

    def itemById(self, command_id):
        return self._defs.get(command_id)


class FakeUI:
    def __init__(self, defs):
        self.commandDefinitions = FakeCommandDefinitions(defs)


# ----------------------------------------------------------------------
# Module loading with a fake adsk
# ----------------------------------------------------------------------


def _module(name: str, **attrs) -> types.ModuleType:
    module = types.ModuleType(name)
    for attr, value in attrs.items():
        setattr(module, attr, value)
    return module


@pytest.fixture(scope="module")
def handler_module():
    if FAKE_PACKAGE in sys.modules:
        return sys.modules[f"{FAKE_PACKAGE}.server.command_handler"]

    design_types = type(
        "DesignTypes",
        (),
        {"ParametricDesignType": PARAMETRIC, "DirectDesignType": DIRECT},
    )
    core = _module("adsk.core")
    fusion = _module("adsk.fusion", DesignTypes=design_types)
    cam = _module("adsk.cam")
    adsk = _module("adsk", core=core, fusion=fusion, cam=cam, doEvents=lambda: None)
    sys.modules.update(
        {"adsk": adsk, "adsk.core": core, "adsk.fusion": fusion, "adsk.cam": cam}
    )
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
def rig(handler_module, monkeypatch):
    """A handler wired to a fake design, with both UI commands stubbed."""

    def factory(design=None, with_undo=True, with_redo=True):
        design = design or FakeDesign()
        defs = {}

        if with_undo:

            def undo_execute():
                design.undo_commands_executed += 1

            defs["UndoCommand"] = FakeCommandDefinition("UndoCommand", undo_execute)
        if with_redo:
            defs["RedoCommand"] = FakeCommandDefinition("RedoCommand", lambda: None)

        handler = handler_module.CommandHandler.__new__(handler_module.CommandHandler)
        handler.ui = FakeUI(defs)
        handler._design = lambda: design
        return design, handler

    return factory


# ----------------------------------------------------------------------
# Parametric: verifiable rollback
# ----------------------------------------------------------------------


class TestParametricUndo:
    def test_rolls_the_marker_back_one_step(self, rig):
        design, handler = rig(FakeDesign(features=2))

        result = handler.undo()

        assert design.marker == 1
        assert result["undone"] is True
        assert result["verified"] is True
        assert result["marker_position"] == 1

    def test_leaves_the_feature_in_the_timeline(self, rig):
        """Rollback suppresses rather than deletes — worth saying out loud."""
        design, handler = rig(FakeDesign(features=2))

        result = handler.undo()

        assert design.timeline_count == 2, "the feature must not be deleted"
        assert "suppressed" in result["note"].lower()
        assert result["timeline_count"] == 2

    def test_does_not_use_the_async_ui_command(self, rig):
        design, handler = rig(FakeDesign(features=2))

        handler.undo()

        assert design.undo_commands_executed == 0, (
            "parametric undo must not fall back to the asynchronous UI command"
        )

    def test_reports_nothing_to_undo_at_the_start_of_history(self, rig):
        design, handler = rig(FakeDesign(features=2, marker=0))

        result = handler.undo()

        assert result["undone"] is False
        assert result["reason"]
        assert result["marker_position"] == 0
        assert design.marker == 0

    def test_rolls_forward_again(self, rig):
        """The rollback is reversible, which is a free redo."""
        design, handler = rig(FakeDesign(features=2))

        handler.undo()
        design.timeline.markerPosition = 2

        assert design.marker == 2
        assert design.marker_writes == [1, 2]

    def test_empty_timeline_reports_nothing_to_undo(self, rig):
        """No features means no undo — not an async command nobody can verify."""
        design, handler = rig(FakeDesign(features=0))

        result = handler.undo()

        assert result["undone"] is False
        assert result["verified"] is True
        assert design.undo_commands_executed == 0


# ----------------------------------------------------------------------
# Direct mode: no synchronous handle, so do not pretend
# ----------------------------------------------------------------------


class TestDirectModeFallback:
    def test_get_scene_info_works_on_a_direct_design(self, rig):
        """get_scene_info must not explode in direct mode.

        It read ``design.timeline`` behind ``hasattr``, but hasattr only swallows
        AttributeError and the property raises RuntimeError — so the tool undo's
        own hint sends agents to was an UNKNOWN error in exactly that mode.
        """
        design, handler = rig(
            FakeDesign(features=0, design_type=DIRECT, timeline_raises=True)
        )
        design.productType = "DesignProductType"
        handler._camera_info = lambda: {}

        result = handler.get_scene_info()

        assert result["timeline_count"] == 0
        assert result["bodies_count"] == 0

    def test_undo_works_when_design_timeline_raises(self, rig):
        """In direct mode ``design.timeline`` raises; undo must not read it.

        Measured in Fusion 2706: the property raises "3 : this is not a
        parametric design", which turned undo into an UNKNOWN error instead of
        the documented issued-but-unverified result.
        """
        design, handler = rig(FakeDesign(design_type=DIRECT, timeline_raises=True))

        result = handler.undo()

        assert result["undone"] == "issued"
        assert result["verified"] is False
        assert design.undo_commands_executed == 1

    def test_reports_issued_but_unverified(self, rig):
        design, handler = rig(FakeDesign(design_type=DIRECT))

        result = handler.undo()

        assert result["undone"] == "issued"
        assert result["verified"] is False
        assert design.undo_commands_executed == 1
        combined = (result.get("note", "") + " ".join(result.get("hints", []))).lower()
        assert "async" in combined or "later" in combined
        assert "get_scene_info" in combined

    def test_missing_undo_command_still_refuses(self, rig):
        design, handler = rig(FakeDesign(design_type=DIRECT), with_undo=False)

        with pytest.raises(RuntimeError, match="not available"):
            handler.undo()

        assert design.undo_commands_executed == 0


# ----------------------------------------------------------------------
# The guard the old code could never fire
# ----------------------------------------------------------------------


class TestDesignTypeGuard:
    def test_aborts_if_rolling_back_switches_the_design_type(self, rig):
        design, handler = rig(FakeDesign(features=2, marker_switches_type=True))

        with pytest.raises(RuntimeError, match="Undo aborted"):
            handler.undo()

        assert design.designType == PARAMETRIC, "design was left in direct mode"
        assert design.marker == 2, "the marker was not restored"
