"""Download the 10 papers referenced in SUMMARY.md from arXiv.

Run this on your host machine (not the Cowork sandbox), e.g.:
    python download_pdfs.py
The script writes PDFs into the same folder it lives in.
"""

from pathlib import Path
from urllib.request import Request, urlopen

PAPERS = [
    ("01_RAMTL_2024_RetrievalAugmented_STL_Mining.pdf",          "2405.14355"),
    ("02_NeuroSTL_2022_NN_STL_Interpretable_Classification.pdf", "2210.01910"),
    ("03_TLINet_2024_Differentiable_NN_TL_Inference.pdf",        "2405.06670"),
    ("04_STLDT_2024_Optimal_STL_Decision_Trees_MILP.pdf",        "2407.21090"),
    ("05_AlphaGen_2023_Synergistic_Formulaic_Alpha_RL.pdf",      "2306.12964"),
    ("06_Alpha2_2024_Logical_Formulaic_Alphas_DRL.pdf",          "2406.16505"),
    ("07_QuantFactorREINFORCE_2024_Variance_Bounded.pdf",        "2409.05144"),
    ("08_AlphaGPT_2023_Human_AI_Alpha_Mining.pdf",               "2308.00016"),
    ("09_AlphaAgent_2025_LLM_Alpha_Decay_Resistant.pdf",         "2502.16789"),
    ("10_AlphaJungle_2025_LLM_MCTS_Factor_Mining.pdf",           "2505.11122"),
]

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; paper-downloader/1.0)"}


def main() -> None:
    here = Path(__file__).resolve().parent
    for fname, arxiv_id in PAPERS:
        out = here / fname
        if out.exists() and out.stat().st_size > 0:
            print(f"[skip] {fname} already exists")
            continue
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        print(f"[get ] {arxiv_id}  ->  {fname}")
        req = Request(url, headers=HEADERS)
        with urlopen(req) as r, open(out, "wb") as f:
            f.write(r.read())
    print("Done.")


if __name__ == "__main__":
    main()
