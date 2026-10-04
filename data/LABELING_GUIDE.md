# Labeling guide: research intents

One utterance, one label: the single best intent for **what the user wants produced**. When two intents seem
plausible, apply the decision rules below in order. All training data is synthetic (templates plus LLM-generated text,
then reviewed). Do not add intents.

## Intents

| Intent | What the user wants | Typical shape |
|---|---|---|
| `FACT_LOOKUP` | One concrete, checkable fact (a date, version, number, name, license, yes/no) | "What license does LangGraph use?" |
| `TECHNOLOGY_COMPARISON` | A comparison of two or more **named technologies** along some dimensions | "Compare LangGraph and CrewAI" |
| `LITERATURE_REVIEW` | Papers, surveys, state of the art, what the research says | "Find recent papers on small speech models" |
| `TECHNICAL_DEEP_DIVE` | How **one** technology/system/algorithm works internally | "How does Raft leader election work?" |
| `COMPETITIVE_INTELLIGENCE` | Companies and products in a market: who competes, positioning, offerings | "Who are Datadog's main competitors?" |
| `MARKET_RESEARCH` | Market size, trends, adoption, pricing, forecasts, with figures | "How big is the vector database market?" |
| `RESEARCH_MONITORING` | An **ongoing** request: monitor, track, keep me updated, watch, notify | "Keep me updated on new LLM eval tools" |
| `GENERAL_RESEARCH` | Open-ended exploration, "tell me about", one-off "latest on X", anything else research-shaped | "Tell me about LangGraph" |

## Decision rules for confusable pairs

1. **`FACT_LOOKUP` vs `GENERAL_RESEARCH`.** If a single short sentence could fully answer it and a source could confirm it, it is
   `FACT_LOOKUP`. If the answer needs explanation, several facts, or breadth ("tell me about", "overview of"), it is `GENERAL_RESEARCH`.
   - "When was PyTorch 2.0 released?" -> `FACT_LOOKUP`; "Tell me about PyTorch 2.0" -> `GENERAL_RESEARCH`.
2. **`TECHNOLOGY_COMPARISON` vs `COMPETITIVE_INTELLIGENCE`.** Comparing technologies/frameworks/libraries/protocols on technical
   dimensions is `TECHNOLOGY_COMPARISON`. Analysing **companies or commercial products** and their market position is
   `COMPETITIVE_INTELLIGENCE`. If the named things are products of companies but the question is technical ("Pinecone vs Qdrant latency"),
   it stays `TECHNOLOGY_COMPARISON`; if it is about strategy, pricing, customers, funding or positioning, it is `COMPETITIVE_INTELLIGENCE`.
3. **`LITERATURE_REVIEW` vs `TECHNICAL_DEEP_DIVE`.** Asking for papers, surveys, studies, citations or "what the research says" is
   `LITERATURE_REVIEW`. Asking how one thing works or is built is `TECHNICAL_DEEP_DIVE`, even if the answer will cite papers.
4. **`RESEARCH_MONITORING` vs one-off "latest".** Only an explicit ongoing intent (monitor, track, watch, keep me updated, alert me,
   follow, "every week", "going forward") is `RESEARCH_MONITORING`. A one-off request such as "What's the latest on X?" or "recent
   developments in X" is `GENERAL_RESEARCH` (confirmed with the project owner).
5. **`MARKET_RESEARCH` vs `COMPETITIVE_INTELLIGENCE`.** Aggregate market/size/trend/forecast/adoption questions are `MARKET_RESEARCH`.
   Questions about specific companies' positions are `COMPETITIVE_INTELLIGENCE`.
6. **`GENERAL_RESEARCH` is the default** for research-shaped requests that fit nothing more specific, including "tell me about X".
   "Tell me about LangGraph" -> `GENERAL_RESEARCH`.

## Style variety required in the data

- Typed style ("Compare LangGraph and CrewAI for multi-agent workflows.") and spoken/STT style (lowercase, no punctuation, fillers:
  "um so can you compare langgraph and crewai").
- Short (3 words) to long (35 words); polite and terse; with and without named entities; mixed technologies, companies and topics.
- Include the difficult cases above in both labels so the boundary is learned.

## hard_test.jsonl

About 60 hand-written ambiguous or multi-intent utterances, each labelled with the single best intent by the rules above.
Never used for training, validation or threshold selection.
