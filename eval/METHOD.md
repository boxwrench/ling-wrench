# Evaluation method

All quality numbers on the model card and in this repository come from the same
locked next-token corpus and scorer. The corpus and the reference are fixed
before a candidate is scored; no release candidate is evaluated on text used for
its calibration.

## Corpus

- 92 windows of 4,096 Ling-3.1 token ids: 376,832 positions, of which 376,740
  have a next-token target (the last position of each window has none).
- Domain split:
  - **46 windows** of agentic coding and tool-use sessions from our own
    workflow. These sessions are held out before calibration; no window, and no
    continuation of a window, appears in the calibration mix.
  - **30 windows** of public chat-style traffic (AtomicChat corpus).
  - **16 windows** of public-domain prose (wikitext-2).
- Windows are raw continuations. The scorer does not apply a chat template; the
  model sees the token ids it would see in a prefill.
- The corpus is stored as token ids only. Private session text is not
  redistributed; the public windows are reproducible from their named public
  sources.

The split is reported as three domains because the calibration is aimed at the
first one. A single overall average would hide the trade-off that the Wrench mix
makes.

## Reference

The reference is the FP8 base model (`inclusionAI/Ling-3.1-flash`, FP8
checkpoint), not another quant. For each window the scorer stores the final
hidden stream from the FP8 model and computes the reference distribution with
the FP8 norm and LM head. Candidate metrics are computed against that same
reference pass, so a candidate is never compared to another candidate as truth.

## Metrics

Let `p` be the FP8 reference next-token distribution at a position and `q` the
candidate's, and let `y` be the actual next token.

- **Plain KL (top-64).** `KL(p || q)` with both distributions restricted to the
  candidate's top-64 tokens and renormalized, averaged over positions and
  windows. This is the 92-window headline KL. It is comparable across candidates
  because every candidate is scored against the same reference pass.
- **Exact KL (6-window prefix).** Full-vocabulary `KL(p || q)` on the first six
  windows (three agentic, one public chat, two prose). This removes the top-64
  restriction and checks that the ranking is not an artifact of the truncation.
- **Predictable-position KL.** The same per-position KL restricted to positions
  where the FP8 reference's top-1 *is* the actual next token. These are the
  positions where a next-token mistake is least ambiguous. The kept share of
  positions is reported with the value; a low kept share means the metric covers
  a small, easy subset.
- **dNLL (actual-token NLL difference).** `NLL_candidate(y) − NLL_FP8(y)`,
  averaged over positions. Negative means the candidate assigns more probability
  to the actual token than the FP8 reference does. Exact full-vocabulary dNLL is
  reported on the 6-window prefix. On the 92-window top-64 dumps, a token outside
  the candidate's top-64 gets a uniform-tail estimate; that value is an upper
  bound and is not used to rank candidates.
- **Top-1 agreement.** Fraction of positions where the candidate's argmax equals
  the FP8 reference's argmax.

Lower is better for KL, predictable-position KL and dNLL; higher is better for
top-1. A candidate can win one metric and lose another; the release text reports
all five rather than selecting one after the fact.

## Uncertainty

- **Paired per-window bootstrap.** The window is the resampling unit. For a pair
  of candidates (or a candidate and its repeat control), draw 10,000 bootstrap
  samples of the 92 windows with replacement, compute the mean paired difference
  on each sample, and report the 2.5th–97.5th percentile interval. Pairing is by
  window, so corpus difficulty cancels.
- **Noise floor.** Every comparison is accompanied by a same-configuration
  repeat: either the control run twice, or a no-op variant such as a different
  prompt chunk size. The floor is the observed spread of that repeat on the same
  metric. A candidate delta is judged against the floor, not against a fixed
  absolute threshold.
- **Decision rule.** If the paired interval for the candidate-minus-control
  difference contains zero, or if the absolute delta is inside the measured
  floor, the result is reported as "not distinguishable" rather than as a win.
  A gate set below the measured floor is invalid and is not used.
- **Status of the current ranking.** The speed benchmark and the quality
  rescore are complete, but the same-configuration repeat floor for this scorer
  is still being measured. The published quality table therefore carries
  **"verdict pending a noise-floor check"** until that value lands. The numbers
  are the measured single-run values; they are not retracted, only marked
  provisional.

## Reproducing a score

The scorer is not part of this release repository; it depends on the stored FP8
hidden stream, which is large and not redistributed. The method above is
sufficient to reproduce the numbers from the FP8 checkpoint and the published
token ids:

1. Run the FP8 model over each window and store the final hidden state.
2. Apply the FP8 norm and head to get `p`.
3. Run the candidate over the same ids and get `q` (top-64 is enough for the
   92-window table; full vocabulary for the 6-window prefix).
4. Compute the five metrics per window, then the paired bootstrap.

The token ids for the public windows are reproducible from the named public
sources; the agentic windows are held out and are represented only by aggregate
numbers.
