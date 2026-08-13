# `data_fetcher/` — futures data band (L0–L2)

Turns the raw local minute-bar trees under `Future_minute_data/` (read-only)
into layered artifacts under `~/AutoLLM_data/futures/`. **No network I/O** —
the old AKshare A-share fetcher lives on, untouched, in `data_fetcher_akshare/`.

Spec: `gp_cta/proposal.md` (memory safety + look-ahead safety are co-equal top
priorities). Shared plumbing: `futures_common/`.

## Stages

| Stage | Module | Output |
|---|---|---|
| L0 ingest | `ingest.py` + `readers/` | one parquet per real contract, `raw/exchange=*/product=*/` |
| L1a calendar | `calendar.py` | inferred session templates, trading days, night exceptions |
| L1b continuous | `continuous.py` | dominant-contract series with **PIT segment-START-anchored** adjustment, roll calendar |
| L1c universe | `universe.py` | listings, era flags, multiplier/tick inference, ADV |
| L2a panel | `panel/align.py` | (T, N) float32 memmaps: channels + traded/limit/roll masks + `bob_index` |
| L2b features | `panel/features.py` | 14 causal features, one memmap each |

## CLI

```bash
python -m data_fetcher ingest     --slice acceptance          # RB,I,TA 2019-2021
python -m data_fetcher calendar   --slice acceptance
python -m data_fetcher continuous --slice acceptance
python -m data_fetcher universe   --slice acceptance
python -m data_fetcher panel      --slice acceptance
python -m data_fetcher features   --slice acceptance
python -m data_fetcher check ingest --slice acceptance        # acceptance assertions
```

Slices come from `configs/defaults.toml`; explicit `--products RB,I --start
YYYY-MM-DD --end YYYY-MM-DD` also works. Every run prints peak RSS and writes
a manifest under `results/runs/<stage>/`.

Key semantics worth knowing:

- **Full-lifetime contracts**: `[start, end]` selects WHICH contracts to
  ingest; a touched contract's parquet always holds its full lifetime, so
  overlapping slice runs never clobber each other.
- **Trading-date rule**: night bars (≥ 21:00) belong to the NEXT trading day;
  post-midnight tails (< 03:00) to the first trading day ≥ their date.
- **Dominance is selected within `[start, end]` only** — outside the ingested
  range the store holds partial contract sets that would elect false dominants.
- The truncation-prefix causality gate is pinned by
  `tests/test_features.py::test_truncation_prefix_gate`.

Tests: `pytest data_fetcher -q` (offline, synthetic fixtures);
`pytest data_fetcher -q --realdata` adds real-tree smoke tests.
