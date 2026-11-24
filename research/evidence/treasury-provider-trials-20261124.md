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

On [six resolved Market Pulse cases](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45495133026), replacing the historical chart input with the authenticated API reproduced all frozen source dates, sample sizes, centers and both sets of quantiles within 1e-8. A [forced chart outage](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45495418723) then exercised the actual fallback on one resolved credit-spread case. Its quantiles matched and it published zero forecasts. This measures source equivalence, not prize performance. The adapter is in [Market Pulse validation](https://github.com/xmjunior/mmegalab-futureeval-bot/commit/f2a09239ee817f82d794f625b19dcc792e2068af).

For the same dated search, Exa's anonymous free search returned the [October 28 FOMC release](https://www.federalreserve.gov/newsevents/pressreleases/monetary20261028a.htm). Tavily returned zero eligible URLs among three candidates. The Exa key separately authenticated on a read-only API call. One retrieval comparison measures source coverage only. Groq authenticated and offered GPT-OSS 120B in the model catalog. Live Groq trials use the independent panel and require a confirmed Free organization.

The default branch's existing Champion control loads the research and recovery revisions separately. It reads the stored provider ledger for settlement. The paid OpenAI experiment is limited to a dated push, first attempt, and an exclusive durable lease. Restoring a deployment requires preserving that ledger along with the observations branch. Never reset a completed or ambiguous lease.

## Operational check

On [main revision c05a242](https://github.com/xmjunior/mmegalab-futureeval-bot/commit/c05a242a01390c8673c7eddee6367bf3a9dbf6bc), the [testing-area smoke test](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45495504226) authenticated, received two hosted Gemma 26B responses and validated one complete unpublished forecast. The [recovery clock](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45488322523) dispatched the [seasonal run](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45496761254) for an unanswered window and queued its [successor](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45496795367). The seasonal run authenticated and recorded zero new receipts after 503, 408 and 429 errors. Production remains on the established hosted routes. The API experiment operates only in the research branch.

## Held-out economic analogue

The [frozen comparison](https://github.com/xmjunior/mmegalab-futureeval-bot/commit/813662ef7d9e70a5cf04065b944e5958796bbd48) scored 184 pairs across 46 monthly forecast origins. The empirical reference had Brier 0.258713 and the volatility-scaled challenger 0.260786. The paired gain was −0.002073 with a 95% moving-block bootstrap interval of [−0.006901, +0.002861]. Eight months improved, 26 tied and 12 worsened. Production keeps the empirical reference wherever that reference was already in use. Each case is in [the full replay](treasury-vintage-results.json), SHA-256 b6cbfed526814ce7a49978cddd040dc9f0a99fae3ab654c7b9e2f01edcc97736. [Execution](https://github.com/xmjunior/mmegalab-futureeval-bot/actions/runs/45495809891).
