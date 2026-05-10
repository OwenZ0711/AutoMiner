#!/usr/bin/env bash
# Download the 10 papers referenced in SUMMARY.md from arXiv.
# Run this on your host machine (not the Cowork sandbox), in the same folder.
set -e
cd "$(dirname "$0")"

declare -a PAPERS=(
  "01_RAMTL_2024_RetrievalAugmented_STL_Mining.pdf|2405.14355"
  "02_NeuroSTL_2022_NN_STL_Interpretable_Classification.pdf|2210.01910"
  "03_TLINet_2024_Differentiable_NN_TL_Inference.pdf|2405.06670"
  "04_STLDT_2024_Optimal_STL_Decision_Trees_MILP.pdf|2407.21090"
  "05_AlphaGen_2023_Synergistic_Formulaic_Alpha_RL.pdf|2306.12964"
  "06_Alpha2_2024_Logical_Formulaic_Alphas_DRL.pdf|2406.16505"
  "07_QuantFactorREINFORCE_2024_Variance_Bounded.pdf|2409.05144"
  "08_AlphaGPT_2023_Human_AI_Alpha_Mining.pdf|2308.00016"
  "09_AlphaAgent_2025_LLM_Alpha_Decay_Resistant.pdf|2502.16789"
  "10_AlphaJungle_2025_LLM_MCTS_Factor_Mining.pdf|2505.11122"
)

for entry in "${PAPERS[@]}"; do
  fname="${entry%%|*}"
  arxiv_id="${entry##*|}"
  if [[ -s "$fname" ]]; then
    echo "[skip] $fname already exists"
    continue
  fi
  echo "[get ] $arxiv_id -> $fname"
  curl -fsSL -A "Mozilla/5.0" -o "$fname" "https://arxiv.org/pdf/${arxiv_id}"
done

echo "Done. Files:"
ls -lh *.pdf
