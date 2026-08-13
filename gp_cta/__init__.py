"""Mining band: lead-lag detection (leadlag/) + warm-start GP (gp/).

Consumes the data band's panel artifacts under ``~/AutoLLM_data/futures/``
exclusively through :mod:`gp_cta.panel_io` — the bands share disk contracts,
never imports. Spec: ``gp_cta/proposal.md``; math: ``docs/LEADLAG_DESIGN.md``.
"""

__version__ = "0.1.0"
