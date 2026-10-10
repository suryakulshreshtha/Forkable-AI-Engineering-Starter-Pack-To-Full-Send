"""The judge for drill 7.2."""
from kit.fakemodel import ScriptedModel
from kit.tools import ToolRegistry


def _tools():
    reg = ToolRegistry()
    reg.register("add", lambda a, b: a + b, {"a": float, "b": float})
    reg.register("lookup", lambda key: {"capital_of_peru": "Lima"}[key], {"key": str})
    return reg


def test_a_direct_answer_takes_one_step(m):
    model = ScriptedModel([{"type": "final", "text": "Hello!"}])
    run = m.run_agent(model, _tools(), "hi")
    assert run["answer"] == "Hello!" and run["steps"] == 1 and run["stopped"] == "final"
    assert run["messages"] == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    assert run["tool_calls"] == []


def test_system_message_goes_first(m):
    model = ScriptedModel([{"type": "final", "text": "ok"}])
    m.run_agent(model, _tools(), "hi", system="Be brief.")
    assert model.seen[0][0] == {"role": "system", "content": "Be brief."}
    assert model.seen[0][1] == {"role": "user", "content": "hi"}


def test_tool_result_is_shown_to_the_model(m):
    model = ScriptedModel([
        {"type": "tool_call", "name": "add", "args": {"a": 2, "b": 40}},
        lambda msgs: {"type": "final", "text": f"The answer is {msgs[-1]['content']['result']}"},
    ])
    run = m.run_agent(model, _tools(), "what is 2 + 40?")
    assert run["answer"] == "The answer is 42"
    assert run["steps"] == 2 and run["tool_calls"] == [("add", True)]
    second_view = model.seen[1]
    assert second_view[-2] == {"role": "assistant", "tool_call": {"name": "add", "args": {"a": 2, "b": 40}}}
    assert second_view[-1] == {"role": "tool", "name": "add", "content": {"ok": True, "result": 42}}


def test_tool_errors_are_fed_back_not_raised(m):
    model = ScriptedModel([
        {"type": "tool_call", "name": "lookup", "args": {"key": "capital_of_mars"}},
        {"type": "tool_call", "name": "lookup", "args": {"key": "capital_of_peru"}},
        lambda msgs: {"type": "final", "text": msgs[-1]["content"]["result"]},
    ])
    run = m.run_agent(model, _tools(), "capital of peru?")
    assert run["answer"] == "Lima"
    assert run["tool_calls"] == [("lookup", False), ("lookup", True)]
    failed = model.seen[1][-1]
    assert failed["role"] == "tool" and failed["content"]["ok"] is False and "KeyError" in failed["content"]["error"]


def test_a_stuck_model_is_cut_off(m):
    model = ScriptedModel([{"type": "tool_call", "name": "add", "args": {"a": 1, "b": 1}}], repeat_last=True)
    run = m.run_agent(model, _tools(), "loop forever", max_steps=4)
    assert run["stopped"] == "max_steps" and run["answer"] is None
    assert model.calls == 4 and run["steps"] == 4 and len(run["tool_calls"]) == 4


def test_invalid_replies_get_a_correction(m):
    model = ScriptedModel([
        {"type": "poem", "text": "roses are red"},
        {"type": "final", "text": "sorry, here is my answer"},
    ])
    run = m.run_agent(model, _tools(), "hi")
    assert run["answer"] == "sorry, here is my answer" and run["steps"] == 2
    note = model.seen[1][-1]
    assert note["role"] == "system" and "final" in note["content"] and "tool_call" in note["content"]


def test_unknown_tools_do_not_crash_the_loop(m):
    model = ScriptedModel([
        {"type": "tool_call", "name": "launch_rockets", "args": {}},
        {"type": "final", "text": "I can't do that."},
    ])
    run = m.run_agent(model, _tools(), "launch")
    assert run["tool_calls"] == [("launch_rockets", False)] and run["stopped"] == "final"
