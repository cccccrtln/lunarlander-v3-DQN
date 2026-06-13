from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLOT_DIR = PROJECT_ROOT / "results" / "round_07_path_d_integrated_dqn" / "plots"
SOLVED_LINE = 200.0

RUNS = {
    "Path A combo C": PROJECT_ROOT
    / "results"
    / "round_04_path_a_vanilla_dqn_tuning"
    / "_single_factor_screening"
    / "cache"
    / "combo__combo_aggressive_64__300000.json",
    "Path B double + Path A params": PROJECT_ROOT
    / "results"
    / "round_05_path_b_dqn_structure_variants"
    / "_cache"
    / "trial_double_dqn_with_patha_comboC__300000__seed11.json",
    "Path C objective shaping (base)": PROJECT_ROOT
    / "results"
    / "round_06_path_c_lunarlander_auxiliary_heads"
    / "_cache"
    / "trial_baseline_plus_objective_shaping__300000__seed11.json",
    "Path D mid_updates": PROJECT_ROOT
    / "results"
    / "round_07_path_d_integrated_dqn"
    / "_cache"
    / "trial_pathD_mid_updates__300000__seed11.json",
}


def load_run(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def final_return(run: dict) -> float:
    return float(run["records"][-1]["eval_mean_return"])


def export_plot(selected: list[tuple[str, dict]], filename: str, title: str) -> Path:
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10.8, 6.0))
    linestyles = ["-", "--", "-.", ":"]
    for idx, (label, run) in enumerate(selected):
        df = pd.DataFrame(run["records"])
        ax.plot(
            df["timesteps"],
            df["eval_mean_return"],
            marker="o",
            linewidth=2.2,
            linestyle=linestyles[idx % len(linestyles)],
            label=label,
        )
    ax.axhline(SOLVED_LINE, color="gray", linestyle="--", linewidth=1.2, label="solved threshold (200)")
    ax.set_title(title)
    ax.set_xlabel("Environment steps")
    ax.set_ylabel("Deterministic eval mean return")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", fontsize=9)
    out = PLOT_DIR / filename
    fig.savefig(out, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> None:
    loaded = [(label, load_run(path)) for label, path in RUNS.items()]
    loaded.sort(key=lambda item: final_return(item[1]), reverse=True)

    top3 = loaded[:3]
    top2 = loaded[:2]

    top3_path = export_plot(
        top3,
        "current_top3_models_by_final_300k.png",
        "Current Top 3 Models by 300k Final Return",
    )
    top2_path = export_plot(
        top2,
        "current_top2_models_by_final_300k.png",
        "Current Top 2 Models by 300k Final Return",
    )

    print(f"Top 3 plot written to: {top3_path}")
    print(f"Top 2 plot written to: {top2_path}")
    print("Ranking:")
    for label, run in loaded:
        print(f"- {label}: final={final_return(run):.2f}")


if __name__ == "__main__":
    main()
