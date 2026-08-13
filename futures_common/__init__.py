"""Shared spine for the futures GP-CTA pipeline.

Leaf package with zero project dependencies. Both bands (`data_fetcher`,
`gp_cta`) import from here; nothing here imports from them. Modules:

- ``paths``    — every on-disk location (data root + results root)
- ``config``   — typed config loaders + ``config_hash``
- ``memory``   — peak-RSS tracking and enforcement (< 4 GB per stage)
- ``manifest`` — per-run manifest (config, seed, git hash, peak RSS)
- ``log``      — loguru setup
- ``rng``      — seeded named-stream RNG factory (bare ``np.random.*`` is banned)
"""

__version__ = "0.1.0"
