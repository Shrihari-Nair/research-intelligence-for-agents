"""Synthetic research-domain utterances: deterministic templates (this file) + LLM-generated rows (llm_generate.py).

All data is synthetic. Labels follow data/LABELING_GUIDE.md.
"""
from __future__ import annotations

import random
import re

import pandas as pd

from research_os.router.data.validate import normalize

TECHS = [
    "LangGraph", "CrewAI", "AutoGen", "LlamaIndex", "LangChain", "PyTorch", "TensorFlow", "JAX", "Kubernetes",
    "Docker Swarm", "Nomad", "Kafka", "Pulsar", "RabbitMQ", "PostgreSQL", "MongoDB", "DuckDB", "SQLite", "Redis",
    "Memcached", "FastAPI", "Flask", "Django", "Rust", "Go", "Zig", "Qdrant", "Pinecone", "Weaviate", "Milvus",
    "pgvector", "Elasticsearch", "OpenSearch", "Terraform", "Pulumi", "GraphQL", "gRPC", "WebAssembly", "vLLM",
    "TensorRT-LLM", "Ollama", "ONNX Runtime", "Ray", "Dask", "Spark", "Airflow", "Prefect", "Dagster", "dbt",
    "React", "Svelte", "Vue", "Next.js", "Tailwind", "Langfuse", "MLflow", "Weights & Biases", "Triton", "Istio",
    "Envoy", "Cassandra", "ClickHouse", "Snowflake", "Delta Lake", "Apache Iceberg", "Hugging Face Transformers",
]
COMPANIES = [
    "OpenAI", "Anthropic", "Google DeepMind", "Mistral AI", "Cohere", "Databricks", "Snowflake", "Datadog",
    "Grafana Labs", "New Relic", "HashiCorp", "Vercel", "Netlify", "Stripe", "Adyen", "Notion", "Figma", "Canva",
    "Palantir", "Scale AI", "Hugging Face", "Nvidia", "AMD", "Cerebras", "Groq", "Pinecone", "Weaviate", "Elastic",
    "MongoDB Inc", "Confluent", "Cloudflare", "Fastly", "Twilio", "Zendesk", "Shopify", "Atlassian", "Slack",
]
TOPICS = [
    "retrieval augmented generation", "small language models", "speech recognition", "vector search",
    "federated learning", "model quantization", "LLM evaluation", "agent memory", "diffusion models", "edge AI",
    "robotics foundation models", "protein structure prediction", "graph neural networks",
    "prompt injection defenses", "multi-agent systems", "mixture of experts", "speculative decoding",
    "differential privacy", "retrieval reranking", "long-context transformers", "reinforcement learning from feedback",
    "synthetic data generation", "knowledge graphs", "time series forecasting", "code generation models",
    "multimodal learning", "neural architecture search", "causal inference", "streaming data pipelines",
    "zero-knowledge proofs", "serverless computing", "observability for LLM apps",
]
MARKETS = [
    "vector database market", "MLOps platform market", "edge AI chip market", "observability tooling market",
    "AI coding assistant market", "conversational AI market", "cloud data warehouse market",
    "enterprise search market", "API management market", "synthetic data market", "speech analytics market",
    "low-code platform market", "cybersecurity SaaS market", "digital twin market", "AI governance software market",
]
DIMS = [
    "latency", "cost", "scalability", "ease of use", "ecosystem", "license", "community support", "memory usage",
    "throughput", "learning curve", "deployment complexity", "observability support", "security model",
    "tooling maturity", "vendor lock-in",
]
YEARS = ["2022", "2023", "2024", "2025", "the last two years", "the past year", "the last five years"]
ATTRS = ["license", "latest stable version", "release date", "creator", "programming language", "default port",
         "maintainer", "first release year", "minimum Python version", "package name on PyPI"]

PREFIX = ["um ", "so ", "uh ", "hey ", "okay ", "hi, ", "quick question, ", "can you ", "could you ", "i need you to ",
          "i'd like you to ", "please ", "so basically ", "right so "]
SUFFIX = [" please", " thanks", " if you can", " as soon as possible", " for a report", " for my team"]

T: dict[str, list[str]] = {
    "FACT_LOOKUP": [
        "what license does {t1} use", "who created {t1}", "when was {t1} first released",
        "what programming language is {t1} written in", "what is the latest stable version of {t1}",
        "is {t1} open source", "what is the default port of {t1}", "which company maintains {t1}",
        "what year did {c1} release {t1}", "how many employees does {c1} have", "when was {c1} founded",
        "where is {c1} headquartered", "who is the CEO of {c1}", "what is the minimum Python version for {t1}",
        "does {t1} support {t2} natively", "what is the package name of {t1} on PyPI",
        "what is the maximum context length of {t1}", "which cloud does {c1} run on",
        "is {t1} licensed under Apache 2.0", "what does the acronym {t1} stand for",
    ],
    "TECHNOLOGY_COMPARISON": [
        "compare {t1} and {t2}", "{t1} vs {t2} for {topic}", "what are the differences between {t1} and {t2}",
        "compare {t1}, {t2} and {t3} on {d1}", "which is better for production, {t1} or {t2}",
        "how does {t1} stack up against {t2} in terms of {d1}", "{t1} versus {t2}: {d1} and {d2}",
        "should we pick {t1} or {t2} for {topic}", "give me a comparison of {t1} and {t2} on {d1} and {d2}",
        "pros and cons of {t1} compared to {t2}", "compare the {d1} of {t1} and {t2}",
        "is {t1} faster than {t2}", "contrast {t1} with {t2} and {t3}", "benchmark {t1} against {t2}",
        "what should I choose between {t1} and {t2} given our focus on {d1}",
        "how do {t1} and {t2} differ in {d1} and {d2}", "evaluate {t1} vs {t2} vs {t3} for a startup",
        "{t1} or {t2}, which one scales better", "side by side comparison of {t1} and {t2}",
        "migrating from {t1} to {t2}: how do they compare",
    ],
    "LITERATURE_REVIEW": [
        "find recent papers on {topic}", "give me a literature review of {topic}",
        "what does the research say about {topic}", "survey the state of the art in {topic}",
        "list key papers on {topic} from {y}", "summarize the main approaches to {topic} in the literature",
        "are there any good survey papers on {topic}", "what are the open problems in {topic} according to recent research",
        "find arxiv papers about {topic} and {topic2}", "review the academic literature on {topic}",
        "which studies have evaluated {topic}", "collect the most cited papers on {topic}",
        "what are the landmark papers in {topic}", "put together a reading list on {topic}",
        "what have researchers found about {topic} in {y}", "compile a systematic review of {topic}",
        "show me papers that compare approaches to {topic}", "which benchmarks are used in {topic} research",
        "who are the leading research groups working on {topic}", "map the research landscape of {topic}",
    ],
    "TECHNICAL_DEEP_DIVE": [
        "how does {t1} work internally", "explain the architecture of {t1}",
        "how does {t1} handle {topic}", "walk me through how {t1} schedules tasks under the hood",
        "what is the internal design of {t1}", "how is {t1} implemented", "deep dive into how {t1} stores data",
        "explain how {t1} achieves its {d1}", "break down the execution model of {t1}",
        "how does {t1} implement {topic} at a low level", "what happens when a request hits {t1}",
        "describe the memory model of {t1}", "how does {t1} manage state across nodes",
        "explain the algorithm behind {topic}", "how does {topic} actually work step by step",
        "what are the internals of {t1}'s query planner", "walk through the source architecture of {t1}",
        "technical breakdown of how {t1} does {topic}", "how does {t1} guarantee consistency",
        "explain the design tradeoffs inside {t1}",
    ],
    "COMPETITIVE_INTELLIGENCE": [
        "who are the main competitors of {c1}", "analyze {c1}'s position against {c2}",
        "how does {c1} compete with {c2} and {c3}", "what is {c1}'s product strategy compared to {c2}",
        "profile {c1} and its top rivals", "what are {c1}'s strengths and weaknesses versus {c2}",
        "map the competitive landscape around {c1}", "how is {c1} differentiating from {c2}",
        "what pricing model does {c1} use compared to {c2}", "which customers does {c1} win against {c2}",
        "competitive analysis of {c1}, {c2} and {c3}", "what has {c1} launched recently to counter {c2}",
        "who competes with {c1} in {market}", "how does {c1}'s go-to-market compare with {c2}",
        "what are {c2}'s offerings that threaten {c1}", "build a competitor profile for {c1}",
        "where is {c1} vulnerable relative to {c2}", "which companies are {c1}'s fastest growing rivals",
        "what partnerships has {c1} announced versus {c2}", "who are the new entrants challenging {c1}",
    ],
    "MARKET_RESEARCH": [
        "how big is the {market}", "what is the projected growth of the {market} through 2030",
        "what are the main trends in the {market}", "who are the biggest players in the {market} by revenue",
        "what is the adoption rate of {topic} in enterprises", "size the {market} for {y}",
        "how fast is the {market} growing", "what are typical prices in the {market}",
        "market forecast for {topic} over the next five years", "what share of companies use {topic} today",
        "which segments of the {market} are growing fastest", "how much funding went into {topic} startups in {y}",
        "what are analysts saying about the {market}", "give me market size figures for {market} with dates",
        "what is the CAGR of the {market}", "how many enterprises have adopted {t1}",
        "what regions lead the {market}", "what are the drivers and risks in the {market}",
        "how has demand for {topic} changed in {y}", "estimate the total addressable market for {topic} tools",
    ],
    "RESEARCH_MONITORING": [
        "keep me updated on {topic}", "monitor new releases of {t1}", "track developments in {topic} for me",
        "alert me when {c1} announces anything new", "watch the {market} and tell me what changes",
        "follow {t1} and notify me of breaking changes", "set up monitoring for new papers on {topic}",
        "keep track of {c1} product launches going forward", "let me know whenever {t1} publishes a new version",
        "i want weekly updates on {topic}", "monitor what {c1} and {c2} announce each month",
        "track every new {topic} paper that appears", "keep an eye on {t1} security advisories",
        "continuously watch for news about {c1}", "send me updates whenever the {market} shifts",
        "subscribe me to developments in {topic}", "follow changes to {t1} release notes from now on",
        "watch {c1}'s changelog and report back regularly", "i'd like ongoing tracking of {topic}",
        "can you monitor {t1} versus {t2} adoption over the coming months",
    ],
    "GENERAL_RESEARCH": [
        "tell me about {t1}", "what is {topic}", "give me an overview of {topic}",
        "what's the latest on {topic}", "recent developments in {t1}", "i want to learn about {topic}",
        "explain {t1} to me", "what should I know about {c1}", "research {topic} for me",
        "tell me about {c1}", "what is going on with {topic} lately", "help me understand {topic}",
        "give me background on {t1}", "what are people saying about {t1}", "find me information about {topic}",
        "i'm curious about {topic}, what can you find", "what's new with {c1}", "summarize what is known about {topic}",
        "give me a primer on {t1}", "look into {topic} and report back once",
    ],
}


def _fill(template: str, rng: random.Random) -> str:
    t1, t2, t3 = rng.sample(TECHS, 3)
    c1, c2, c3 = rng.sample(COMPANIES, 3)
    d1, d2 = rng.sample(DIMS, 2)
    topic, topic2 = rng.sample(TOPICS, 2)
    return template.format(t1=t1, t2=t2, t3=t3, c1=c1, c2=c2, c3=c3, d1=d1, d2=d2, topic=topic, topic2=topic2,
                           market=rng.choice(MARKETS), y=rng.choice(YEARS), attr=rng.choice(ATTRS))


def style(text: str, rng: random.Random) -> str:
    """Speech-style fillers and STT-style lowercase/no punctuation, or typed capitalization."""
    if rng.random() < 0.30:
        text = rng.choice(PREFIX) + text
    if rng.random() < 0.15:
        text += rng.choice(SUFFIX)
    r = rng.random()
    if r < 0.35:  # STT-like
        return re.sub(r"[^\w\s'&+.-]", "", text.lower()).strip()
    text = text[0].upper() + text[1:]
    if rng.random() < 0.6 and text[-1] not in "?.!":
        text += "?" if text.split()[0].lower() in {"what", "how", "who", "when", "where", "which", "is", "does", "are",
                                                      "can", "could"} else "."
    return text


def word_ok(t: str) -> bool:
    """3-35 words."""
    return 3 <= len(t.split()) <= 35


def generate_templates(seed: int, per_intent: int) -> pd.DataFrame:
    """`per_intent` unique template-derived rows per intent (deterministic for a seed)."""
    rng = random.Random(seed)
    rows: list[tuple[str, str]] = []
    for intent, templates in T.items():
        seen: set[str] = set()
        out: list[str] = []
        attempts = 0
        while len(out) < per_intent:
            attempts += 1
            if attempts > 100_000:
                raise RuntimeError(f"could not reach {per_intent} unique rows for {intent}")
            text = style(_fill(rng.choice(templates), rng), rng)
            n = normalize(text)
            if n in seen or not word_ok(text):
                continue
            seen.add(n)
            out.append(text)
        rows += [(t, intent) for t in out]
    return pd.DataFrame(rows, columns=["text", "intent"])
