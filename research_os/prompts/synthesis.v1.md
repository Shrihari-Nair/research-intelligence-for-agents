You write the findings section of a research report for: {request}

Objective: {objective}

You are given VERIFIED claims only. Every finding you write must be derived from one or more of these claims and must cite their claim ids. Do not add facts, numbers, comparisons or causal statements that the claims do not contain. Do not mention anything that is not in the claims.

Verified claims (id: statement):
{claims}

Return ONLY JSON: {{"summary": "<2-4 sentences using only the claims>", "findings": [{{"statement": "<one finding>", "claim_ids": ["<id>"]}}]}}
Every finding needs at least one claim id from the list above.
