"""Usage: python -m research_os.legacy.demo "Can I move my appointment to Friday?" """
from __future__ import annotations

import sys

from research_os.common import setup_logging
from research_os.legacy.factory import build_app_graph
from research_os.legacy.graph.workflow import run_text


def main() -> None:
    setup_logging()
    text = " ".join(sys.argv[1:]).strip() or "Can I move my appointment to Friday?"
    graph, _, backend = build_app_graph()
    s = run_text(graph, text)
    r = s["routing"]
    print(f"Utterance:   {text}")
    print(f"Intent:      {r['intent']}  (confidence {r['confidence']:.2f}, source {r['source']}, "
          f"llm_called {r['llm_called']}, llm backend {backend})")
    print(f"Agent:       {s['agent_name']}")
    print(f"Tool calls:  {s['tool_calls']}")
    print(f"Response:    {s['response']}")
    print(f"Latency:     {r['latency_ms']:.1f} ms (routing)")


if __name__ == "__main__":
    main()
