import pytest

from research_os.agents.supervisor import Supervisor
from research_os.contracts import FailureType, RouterOutput, RoutingMode
from research_os.graph.workflow import build_graph, initial_state
from research_os.harness.context import RunContext
from research_os.llm.mock_client import ScriptedLLM
from research_os.observability.tracing import LangfuseTracer, NoopTracer, attach_tracing, make_tracer, run_scores
from research_os.tools import build_registry
from research_os.tools.registry import ToolError


class FakeObs:
    """Records the calls a real Langfuse observation receives (same keyword signatures as the 4.x SDK)."""

    def __init__(self, client, name, as_type, parent, **kw):
        self.client, self.name, self.as_type, self.parent, self.kw = client, name, as_type, parent, kw
        self.trace_id = "trace123"
        self.children, self.events, self.scores, self.ended, self.updates = [], [], [], False, []

    def start_observation(self, *, name, as_type="span", **kw):
        if self.client.fail:
            raise RuntimeError("langfuse down")
        o = FakeObs(self.client, name, as_type, self, **kw)
        self.children.append(o)
        return o

    def update(self, **kw):
        self.updates.append(kw)
        return self

    def end(self, **kw):
        self.ended = True

    def create_event(self, *, name, **kw):
        self.events.append((name, kw))

    def score_trace(self, *, name, value, **kw):
        self.client.scores.append((name, value))


class FakeLangfuse:
    def __init__(self, fail=False):
        self.roots, self.scores, self.flushed, self.fail = [], [], 0, fail

    def start_observation(self, *, name, as_type="span", **kw):
        if self.fail:
            raise RuntimeError("langfuse down")
        o = FakeObs(self, name, as_type, None, **kw)
        self.roots.append(o)
        return o

    def flush(self):
        self.flushed += 1

    def get_trace_url(self, trace_id=None):
        if trace_id is None:                                   # the real SDK needs an active span otherwise (it is ended by then)
            return None
        return f"http://langfuse.example/trace/{trace_id}"


class R:
    def route(self, text):
        return RouterOutput(intent="FACT_LOOKUP", confidence=0.9, routing_mode=RoutingMode.DIRECT, workflow="fact_lookup", latency_ms=1,
                            entities={"technologies": [], "companies": [], "topics": [], "time_scope": []})


class Pipeline:
    """Search agent calls a tool; evidence agent calls the LLM; the verifier reports claims."""

    def run(self, step, inputs, ctx, request):
        if step.agent == "search_agent":
            ctx.tools.call("web_search", {"query": "langgraph"}, ctx, agent="search_agent")
        if step.agent == "evidence_agent":
            ctx.llm.complete("p", meta={"prompt": "evidence.v2"})
        if step.agent == "verifier":
            return {"claims": [{"claim_id": "c", "verdict": "SUPPORTED"}, {"claim_id": "d", "verdict": "UNSUPPORTED"}],
                    "conflicts": [], "gate": {"result": "PASS", "reason": None, "failure_type": None}, "stats": {"supported": 1, "unsupported": 1}}
        return {"ok": step.id}


def traced_run(client, agents=None):
    ctx = RunContext.create("t1", {}, llm=ScriptedLLM(["{}"] * 5), tools=build_registry(mode="fixture"))
    tracer = LangfuseTracer(client)
    attach_tracing(ctx, tracer, "what license does langgraph use")
    ag = agents or {n: Pipeline() for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}
    out = build_graph(R(), Supervisor(), ctx, agents=ag).invoke(initial_state("what license does langgraph use"))
    ctx.events.emit("run", "finished", {"status": out["status"]})
    return out, ctx, tracer


def names(o):
    return [c.name for c in o.children]


def test_span_tree_has_router_supervisor_planner_steps_with_nested_tool_and_llm_spans():
    lf = FakeLangfuse()
    out, _, tracer = traced_run(lf)
    assert out["status"] == "SUCCESS" and len(lf.roots) == 1
    root = lf.roots[0]
    assert root.name == "research_run" and names(root)[:3] == ["router", "supervisor", "planner"]
    steps = {c.name: c for c in root.children if c.name.startswith("step:")}
    assert list(steps) == ["step:search", "step:read", "step:extract", "step:verify", "step:synthesize"]
    tool = steps["step:search"].children[0]
    assert tool.name == "tool:web_search" and tool.as_type == "tool" and tool.ended
    gen = steps["step:extract"].children[0]
    assert gen.as_type == "generation" and gen.kw.get("model") is not None and gen.ended
    assert gen.kw["metadata"]["prompt"] == "evidence.v2"
    usage = [u for u in gen.updates if "usage_details" in u][0]["usage_details"]
    assert usage["input"] > 0 and "output" in usage
    assert steps["step:verify"].as_type == "evaluator" and steps["step:search"].as_type == "agent"
    assert all(c.ended for c in root.children) and not root.ended      # the root stays open until the service ends the run
    tracer.end_run("SUCCESS", {})
    assert root.ended


def test_failures_are_marked_error_and_recovery_decisions_become_events():
    lf = FakeLangfuse()

    class Flaky(Pipeline):
        n = 0

        def run(self, step, inputs, ctx, request):
            if step.agent == "source_agent" and Flaky.n == 0:
                Flaky.n += 1
                raise ToolError(FailureType.TOOL_FAILURE, "boom")
            return super().run(step, inputs, ctx, request)
    Flaky.n = 0
    from research_os.harness.recovery import RecoveryManager
    ctx = RunContext.create("t2", {}, llm=ScriptedLLM(["{}"] * 5), tools=build_registry(mode="fixture"))
    attach_tracing(ctx, LangfuseTracer(lf), "q")
    ag = {n: Flaky() for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}
    out = build_graph(R(), Supervisor(), ctx, agents=ag, recovery=RecoveryManager()).invoke(initial_state("q"))
    assert out["status"] == "SUCCESS"
    root = lf.roots[0]
    reads = [c for c in root.children if c.name == "step:read"]
    assert len(reads) == 2 and reads[0].updates[-1]["level"] == "ERROR" and "boom" in reads[0].updates[-1]["status_message"]
    assert any(n.startswith("recovery") and kw["metadata"]["action"] == "retry" for n, kw in root.events)


def test_scores_are_attached_at_run_end_and_flushed():
    lf = FakeLangfuse()
    out, ctx, tracer = traced_run(lf)
    tracer.end_run("SUCCESS", run_scores(out, ctx))
    scores = dict(lf.scores)
    assert scores["supported_claims"] == 1 and scores["unsupported_claim_rate"] == 0.5 and scores["run_success"] == 1.0
    assert scores["steps_used"] == 5 and lf.flushed >= 1


def test_run_scores_from_state():
    class S:
        def __init__(self, a, o):
            self.agent, self.outputs = a, o
    class P:
        steps = [S("verifier", {"claims": [{"verdict": "SUPPORTED"}] * 3 + [{"verdict": "UNSUPPORTED"}], "conflicts": [1, 2]})]
    ctx = RunContext.create("s", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
    sc = run_scores({"plan": P(), "status": "PARTIAL"}, ctx)
    assert sc["unsupported_claim_rate"] == 0.25 and sc["conflicts"] == 2.0 and sc["run_success"] == 0.0


def test_tracing_is_a_noop_without_keys_and_adds_no_listener():
    assert isinstance(make_tracer({}), NoopTracer)
    assert isinstance(make_tracer({"LANGFUSE_PUBLIC_KEY": "pk"}), NoopTracer)       # a secret key is required too
    ctx = RunContext.create("n", {}, llm=ScriptedLLM([]), tools=build_registry(mode="fixture"))
    attach_tracing(ctx, NoopTracer(), "q")
    assert ctx.events._listeners == []


def test_tracing_failures_never_break_a_run():
    lf = FakeLangfuse(fail=True)
    out, _, tracer = traced_run(lf)
    assert out["status"] == "SUCCESS"
    tracer.end_run("SUCCESS", {"x": 1.0})                  # must not raise either


def test_keys_enable_the_real_langfuse_tracer_with_host_from_env():
    made = {}

    def factory(**kw):
        made.update(kw)
        return FakeLangfuse()
    t = make_tracer({"LANGFUSE_PUBLIC_KEY": "pk", "LANGFUSE_SECRET_KEY": "sk", "LANGFUSE_HOST": "http://lf:3000"}, client_factory=factory)
    assert isinstance(t, LangfuseTracer) and made["public_key"] == "pk" and made["base_url"] == "http://lf:3000"


def test_real_sdk_accepts_our_calls_offline_without_raising(monkeypatch):
    """Smoke test against the installed Langfuse SDK (4.x) with an unreachable host: no exception, no hang."""
    pytest.importorskip("langfuse")
    t = make_tracer({"LANGFUSE_PUBLIC_KEY": "pk-test", "LANGFUSE_SECRET_KEY": "sk-test", "LANGFUSE_HOST": "http://127.0.0.1:9"})
    assert isinstance(t, LangfuseTracer)
    ctx = RunContext.create("real", {}, llm=ScriptedLLM(["{}"] * 5), tools=build_registry(mode="fixture"))
    attach_tracing(ctx, t, "q")
    ag = {n: Pipeline() for n in ("search_agent", "source_agent", "evidence_agent", "verifier", "synthesis_agent")}
    out = build_graph(R(), Supervisor(), ctx, agents=ag).invoke(initial_state("q"))
    t.end_run("SUCCESS", run_scores(out, ctx))
    assert out["status"] == "SUCCESS"


def test_trace_url_is_built_from_the_root_trace_id_after_the_root_has_ended():
    lf = FakeLangfuse()
    _, _, tracer = traced_run(lf)
    tracer.end_run("SUCCESS", {})
    assert tracer.trace_url == "http://langfuse.example/trace/trace123"
