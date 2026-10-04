# Verifier gold-set evaluation: how to read these numbers

- `verifier_eval_run1_prerevision.json`: first run on the 30 hand-written pairs: **86.7% (26/30)**. Errors: g06 and g08 (the
  claims contained details absent from their quotes, i.e. label problems), and g25/g26 (a real design flaw: the deterministic
  number rule rejected "400 vs 40 participants" as UNSUPPORTED instead of letting the judge say CONTRADICTED).
- Fixes: the number rule now rejects only when the quote contains no numbers at all; the two g06/g08 claims were reworded so they
  are entailed by their quotes (noted per row in `evaluation/dataset/verifier_gold.jsonl`).
- `verifier_eval.json`: second run after those fixes: **30/30**. Because the set was revised after seeing run 1, this is **not an
  unbiased estimate**. The labels were written by the project author, the set is small, and it mostly contains clean cases.
  Treat it as a regression test of the verifier plus an upper bound, not as a measured accuracy.
