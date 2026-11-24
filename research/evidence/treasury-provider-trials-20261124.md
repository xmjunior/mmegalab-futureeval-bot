# Treasury panel and source comparison · 24 November 2026

[Four prospective Treasury questions](../promo/panel.json) use common evidence through 23 November. Each probability was sealed before observation. The December 2 and December 9 dates yield two correlated pairs. Scoring uses official Treasury data and the competition-independent settlement rule. The earliest scoring dates are December 4 and December 11. Promotion is disabled for this panel.

| Forecast | Dec 2 · 2Y | Dec 2 · spread | Dec 9 · 2Y | Dec 9 · spread |
|---|---:|---:|---:|---:|
| Empirical reference as displayed in the prompt | .514286 | .314286 | .500000 | .363636 |
| GPT-5.6 Sol API | .514286 | .314286 | .500000 | .363636 |
| Gemini 4 Neon web | .50 | .33 | .49 | .38 |
| Codex session, GPT-6 family | .49 | .34 | .49 | .38 |

GPT-5.6 Sol returned the four reference probabilities verbatim. Its [one completed API call](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45493080434) recorded 512 uncached input and 166 output tokens, including 90 reasoning tokens. Standard usage cost was USD 0.005368. Account statement reconciliation is distinct from metering. The immutable [provider receipt](https://github.com/xmjunior/mmegalab-futureeval-bot/blob/ad6ccf32b1ec86e5f6a5d95489bd0491b0fe948c/promo-ledger.json) and [paired panel](https://github.com/xmjunior/mmegalab-futureeval-bot/blob/e87d1f21c9900c0931b72d50c77b0bc3e4e82528/records/promo-score.json) can be checked without a provider key.

The [source comparison](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45492682677) found 223 observations each for ALFRED DGS2 and DGS10 matching the frozen Treasury source exactly. The 23 November vintage ended on 20 November. Keep the Treasury source for the missing 23 November observation. The research pipeline can use vintage-dated ALFRED SOFR if the New York Fed is unavailable.

For the same dated search, Exa's anonymous free search returned the [October 28 FOMC release](https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm). Tavily returned zero eligible URLs among three candidates. The Exa key separately authenticated on a read-only API call. One retrieval comparison measures source coverage only. Groq authenticated and offered GPT-OSS 120B in the model catalog. Live Groq trials use the independent panel and require a confirmed Free organization.

The default branch's existing Champion control loads the research and recovery revisions separately. It reads the stored provider ledger for settlement. The paid OpenAI experiment is limited to a dated push, first attempt, and an exclusive durable lease. Restoring a deployment requires preserving that ledger along with the observations branch. Never reset a completed or ambiguous lease.
