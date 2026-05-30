"""Unit tests for core.state — SlotsBase, WorkflowState persistence."""
from dataclasses import dataclass

from core.state import SlotsBase, WorkflowState


@dataclass
class FakeSlots(SlotsBase):
    a: str | None = None
    b: str | None = None
    c: str | None = None

    def required_keys(self) -> list[str]:
        return ["a", "b"]


def test_slots_filled_required():
    s = FakeSlots(a="x", b=None, c="z")
    assert s.filled_required() == 1
    assert s.completeness() == 0.5


def test_slots_all_filled():
    s = FakeSlots(a="x", b="y", c="z")
    assert s.filled_required() == 2
    assert s.completeness() == 1.0


def test_slots_get_set():
    s = FakeSlots()
    s.set("a", "value")
    assert s.get("a") == "value"
    s.set("nonexistent", "ignored")
    assert s.get("nonexistent") is None


def test_slots_from_dict_ignores_unknown():
    s = FakeSlots.from_dict({"a": "x", "unknown_key": "ignored"})
    assert s.a == "x"
    assert s.b is None


def test_workflow_state_save_load_roundtrip(tmp_path):
    path = tmp_path / "sess.json"
    st = WorkflowState(
        session_id="sess",
        workflow_name="test-workflow",
        slots=FakeSlots(a="hello"),
    )
    st.iteration = 5
    st.add_user_msg("hi")
    st.extras["motivation_answered"] = True
    st.save(path)

    loaded = WorkflowState.load(path, FakeSlots, "test-workflow")
    assert loaded.iteration == 5
    assert loaded.slots.a == "hello"
    assert loaded.user_history == ["hi"]
    assert loaded.extras["motivation_answered"] is True


def test_workflow_state_load_missing_returns_fresh(tmp_path):
    path = tmp_path / "new.json"
    st = WorkflowState.load(path, FakeSlots, "test-workflow")
    assert st.iteration == 0
    assert st.session_id == "new"
    assert st.workflow_name == "test-workflow"
    assert st.slots.a is None


def test_workflow_state_load_mismatched_resets(tmp_path):
    path = tmp_path / "sess.json"
    st1 = WorkflowState(session_id="sess", workflow_name="wf-A", slots=FakeSlots(a="x"))
    st1.save(path)
    st2 = WorkflowState.load(path, FakeSlots, "wf-B")
    assert st2.workflow_name == "wf-B"
    assert st2.slots.a is None
    assert any("replaced stale state" in line for line in st2.action_log)


def test_workflow_state_summary_shape():
    st = WorkflowState(session_id="s", workflow_name="t", slots=FakeSlots(a="x"))
    st.iteration = 3
    summary = st.summary()
    assert summary["phase"] == "INIT"
    assert summary["iteration"] == 3
    assert summary["slots"]["a"] == "x"
    assert summary["completeness"] == 0.5  # a filled, b not
    assert "extras" in summary


def test_workflow_state_back_compat_skill_name_field(tmp_path):
    """Legacy state files persisted by cyclic-engine v0 have `skill_name`
    instead of `workflow_name`. Loader must accept the older field."""
    import json
    path = tmp_path / "legacy.json"
    legacy = {
        "session_id": "legacy",
        "skill_name": "old-skill",
        "version": 1,
        "created_ts": 0.0,
        "updated_ts": 0.0,
        "iteration": 2,
        "phase": "ASK",
        "slots": {"a": "kept"},
        "user_history": [],
        "bot_history": [],
        "contradictions": [],
        "action_log": [],
        "extras": {},
    }
    path.write_text(json.dumps(legacy), encoding="utf-8")
    loaded = WorkflowState.load(path, FakeSlots, "old-skill")
    assert loaded.workflow_name == "old-skill"
    assert loaded.iteration == 2
    assert loaded.slots.a == "kept"
