# Independent router test set: review sheet

104 queries written by hand (13 per intent) separately from the training-data generator, labelled with `data/LABELING_GUIDE.md` by a **single labeller** (the AI assistant that built this repo).
They are used only to *report* router accuracy (never for training, validation or threshold selection).

**Please spot-check at least 20 rows** (every 5th row is marked ✱). If you disagree with a label, write the intent you would choose in the last column and tell me; I will report the number both ways.

| # | review | text | label | your label |
|---|---|---|---|---|
| 1 |  | how many tokens can GPT-4 Turbo take as input | FACT_LOOKUP | |
| 2 |  | when did the EU AI Act come into force | FACT_LOOKUP | |
| 3 |  | Which programming language is the Linux kernel mostly written in? | FACT_LOOKUP | |
| 4 |  | um what's the latest stable version of postgres right now | FACT_LOOKUP | |
| 5 | ✱ | Is Kubernetes a CNCF graduated project? | FACT_LOOKUP | |
| 6 |  | who is the current CEO of Nvidia | FACT_LOOKUP | |
| 7 |  | How many parameters does Llama 2 70B have? | FACT_LOOKUP | |
| 8 |  | what year was the transformer paper published | FACT_LOOKUP | |
| 9 |  | What is the default TTL for a DNS record in AWS Route 53? | FACT_LOOKUP | |
| 10 | ✱ | does SQLite support full text search natively | FACT_LOOKUP | |
| 11 |  | Which company acquired GitHub in 2018? | FACT_LOOKUP | |
| 12 |  | hey quick question what port does https use by default | FACT_LOOKUP | |
| 13 |  | What is the speed of light in a vacuum in metres per second? | FACT_LOOKUP | |
| 14 |  | Kafka versus RabbitMQ for event streaming, which fits better? | TECHNOLOGY_COMPARISON | |
| 15 | ✱ | compare pytorch lightning and hugging face trainer for finetuning | TECHNOLOGY_COMPARISON | |
| 16 |  | How does gRPC stack up against REST for internal microservice calls? | TECHNOLOGY_COMPARISON | |
| 17 |  | i need to pick between postgres and mongodb for a analytics workload what are the tradeoffs | TECHNOLOGY_COMPARISON | |
| 18 |  | Redis vs Memcached: differences in persistence and eviction | TECHNOLOGY_COMPARISON | |
| 19 |  | Compare Rust and Go for building command line tools. | TECHNOLOGY_COMPARISON | |
| 20 | ✱ | so um terraform or pulumi for infrastructure as code | TECHNOLOGY_COMPARISON | |
| 21 |  | FAISS against Milvus for vector search at ten million vectors | TECHNOLOGY_COMPARISON | |
| 22 |  | What are the differences between OAuth 2.0 and OpenID Connect? | TECHNOLOGY_COMPARISON | |
| 23 |  | tensorflow lite vs onnx runtime on mobile | TECHNOLOGY_COMPARISON | |
| 24 |  | Next.js or Remix for a content heavy site? | TECHNOLOGY_COMPARISON | |
| 25 | ✱ | Compare WebSockets, server sent events, and long polling for live updates | TECHNOLOGY_COMPARISON | |
| 26 |  | can you line up Airflow Dagster and Prefect side by side | TECHNOLOGY_COMPARISON | |
| 27 |  | Find recent papers on retrieval augmented generation evaluation | LITERATURE_REVIEW | |
| 28 |  | what does the research say about intermittent fasting and longevity | LITERATURE_REVIEW | |
| 29 |  | Survey the literature on federated learning for medical imaging | LITERATURE_REVIEW | |
| 30 | ✱ | i need a few good review articles about graph neural networks | LITERATURE_REVIEW | |
| 31 |  | Summarize the state of the art in speech emotion recognition | LITERATURE_REVIEW | |
| 32 |  | What studies exist on the effect of remote work on productivity? | LITERATURE_REVIEW | |
| 33 |  | pull together the key papers on diffusion model distillation | LITERATURE_REVIEW | |
| 34 |  | Give me a literature overview of quantum error correction | LITERATURE_REVIEW | |
| 35 | ✱ | are there any meta analyses on microplastics and human health | LITERATURE_REVIEW | |
| 36 |  | Which papers introduced low rank adaptation and what followed them? | LITERATURE_REVIEW | |
| 37 |  | show me the recent academic work on hallucination detection in language models | LITERATURE_REVIEW | |
| 38 |  | Review the research on sleep deprivation and memory consolidation. | LITERATURE_REVIEW | |
| 39 |  | um what has been published lately about solid state batteries | LITERATURE_REVIEW | |
| 40 | ✱ | How does the Raft consensus algorithm handle leader election? | TECHNICAL_DEEP_DIVE | |
| 41 |  | explain how flash attention reduces memory usage | TECHNICAL_DEEP_DIVE | |
| 42 |  | Walk me through how a B-tree index is stored and searched in PostgreSQL. | TECHNICAL_DEEP_DIVE | |
| 43 |  | how does git store objects internally | TECHNICAL_DEEP_DIVE | |
| 44 |  | What happens inside the JVM when garbage collection runs with G1? | TECHNICAL_DEEP_DIVE | |
| 45 | ✱ | so how does tcp congestion control actually work | TECHNICAL_DEEP_DIVE | |
| 46 |  | Explain how mixture of experts routing works in modern language models. | TECHNICAL_DEEP_DIVE | |
| 47 |  | How does Kubernetes scheduling decide which node a pod lands on? | TECHNICAL_DEEP_DIVE | |
| 48 |  | can you break down how the bloom filter in cassandra works | TECHNICAL_DEEP_DIVE | |
| 49 |  | Describe the internals of the Linux completely fair scheduler. | TECHNICAL_DEEP_DIVE | |
| 50 | ✱ | how do transformers handle positional information | TECHNICAL_DEEP_DIVE | |
| 51 |  | How does Rust's borrow checker enforce memory safety at compile time? | TECHNICAL_DEEP_DIVE | |
| 52 |  | um how does public key infrastructure verify a certificate chain | TECHNICAL_DEEP_DIVE | |
| 53 |  | which vendors does Snowflake mostly lose deals to | COMPETITIVE_INTELLIGENCE | |
| 54 |  | how does Notion position itself against Confluence and Coda | COMPETITIVE_INTELLIGENCE | |
| 55 | ✱ | Analyze Stripe's competitive landscape in online payments. | COMPETITIVE_INTELLIGENCE | |
| 56 |  | what are Cloudflare's main rivals and where do they differ in offering | COMPETITIVE_INTELLIGENCE | |
| 57 |  | Which companies compete with Databricks in the lakehouse space? | COMPETITIVE_INTELLIGENCE | |
| 58 |  | Break down Figma's competitors and their pricing strategy | COMPETITIVE_INTELLIGENCE | |
| 59 |  | um who is going after Zoom's market share these days | COMPETITIVE_INTELLIGENCE | |
| 60 | ✱ | Compare the product positioning of Shopify, BigCommerce and WooCommerce. | COMPETITIVE_INTELLIGENCE | |
| 61 |  | what is Palantir's competitive advantage over other analytics vendors | COMPETITIVE_INTELLIGENCE | |
| 62 |  | Map the competitors of OpenAI by product and target customer. | COMPETITIVE_INTELLIGENCE | |
| 63 |  | Who are Canva's biggest threats? | COMPETITIVE_INTELLIGENCE | |
| 64 |  | How do Slack and Microsoft Teams differentiate their enterprise offerings? | COMPETITIVE_INTELLIGENCE | |
| 65 | ✱ | tell me who competes with Tesla in the autonomous driving space and what they sell | COMPETITIVE_INTELLIGENCE | |
| 66 |  | How large is the global cybersecurity market and how fast is it growing? | MARKET_RESEARCH | |
| 67 |  | what is the forecast for the electric vehicle charging market through 2030 | MARKET_RESEARCH | |
| 68 |  | Estimate the size of the generative AI software market. | MARKET_RESEARCH | |
| 69 |  | um how much do companies spend on cloud infrastructure each year | MARKET_RESEARCH | |
| 70 | ✱ | What are the adoption trends for low code platforms in enterprises? | MARKET_RESEARCH | |
| 71 |  | Give me market size figures for the telehealth industry | MARKET_RESEARCH | |
| 72 |  | what is the projected cagr of the edge computing market | MARKET_RESEARCH | |
| 73 |  | How big is the market for wearable health devices? | MARKET_RESEARCH | |
| 74 |  | Pricing trends in the vector database market over the last two years | MARKET_RESEARCH | |
| 75 | ✱ | what share of enterprises use kubernetes in production | MARKET_RESEARCH | |
| 76 |  | Size and growth of the global e-commerce logistics market? | MARKET_RESEARCH | |
| 77 |  | How fast is the synthetic data market expanding? | MARKET_RESEARCH | |
| 78 |  | so what's the total addressable market for voice assistants in cars | MARKET_RESEARCH | |
| 79 |  | Keep an eye on new releases of the LangChain library and tell me when something changes | RESEARCH_MONITORING | |
| 80 | ✱ | monitor arxiv for new papers on mixture of experts | RESEARCH_MONITORING | |
| 81 |  | Notify me whenever Anthropic publishes a new model announcement. | RESEARCH_MONITORING | |
| 82 |  | um can you track funding rounds for AI chip startups going forward | RESEARCH_MONITORING | |
| 83 |  | Watch the Rust release notes and alert me to breaking changes | RESEARCH_MONITORING | |
| 84 |  | Every week send me updates on new vector database benchmarks | RESEARCH_MONITORING | |
| 85 | ✱ | follow news about the EU data act for me | RESEARCH_MONITORING | |
| 86 |  | Set up tracking for new open source LLM evaluation frameworks | RESEARCH_MONITORING | |
| 87 |  | let me know when there are new CVEs affecting FastAPI | RESEARCH_MONITORING | |
| 88 |  | Keep monitoring competitor pricing pages for observability tools and flag changes | RESEARCH_MONITORING | |
| 89 |  | please keep me posted on developments in nuclear fusion startups | RESEARCH_MONITORING | |
| 90 | ✱ | Track mentions of our product category on Hacker News each day | RESEARCH_MONITORING | |
| 91 |  | alert me if new regulation on facial recognition gets proposed in Europe | RESEARCH_MONITORING | |
| 92 |  | Tell me about the history of the Transformer architecture | GENERAL_RESEARCH | |
| 93 |  | what's the latest on quantum computing | GENERAL_RESEARCH | |
| 94 |  | Give me an overview of zero trust security. | GENERAL_RESEARCH | |
| 95 | ✱ | i want to learn about retrieval augmented generation where do i start | GENERAL_RESEARCH | |
| 96 |  | What is going on with open source large language models lately? | GENERAL_RESEARCH | |
| 97 |  | Research the impact of tariffs on semiconductor supply chains. | GENERAL_RESEARCH | |
| 98 |  | um tell me everything about WebAssembly | GENERAL_RESEARCH | |
| 99 |  | Explore the pros and cons of microservices architecture in general | GENERAL_RESEARCH | |
| 100 | ✱ | What should I know about edge AI? | GENERAL_RESEARCH | |
| 101 |  | recent developments in battery recycling | GENERAL_RESEARCH | |
| 102 |  | Help me understand the landscape of AI regulation worldwide. | GENERAL_RESEARCH | |
| 103 |  | overview of homomorphic encryption and where it is used | GENERAL_RESEARCH | |
| 104 |  | tell me about the rise of small language models | GENERAL_RESEARCH | |
