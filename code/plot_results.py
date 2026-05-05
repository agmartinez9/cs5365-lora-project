"""
plot_results.py
CS 5365 – Deep Learning | University of Texas at El Paso
Group Members: Veronica Aragon, Angela Martinez
Reads results JSON files from the results/ directory and generates
figures comparing model performance across LoRA ranks.
Usage:
    python code/plot_results.py
"""
import json
import os
import matplotlib.pyplot as plt
# ── Configuration ──────────────────────────────────────────────────────────────
RESULTS_DIR = "results"
RANKS = [1, 2, 4, 8, 16, 64]
TASKS = {
    "sst2": {"metric": "eval_accuracy", "label": "Accuracy", "color": "#185FA5"},
    "mrpc": {"metric": "eval_f1",       "label": "F1 Score",  "color": "#0F6E56"},
}
# ── Load Results ───────────────────────────────────────────────────────────────

def load_results(task):
    scores = []
    for rank in RANKS:
        path = os.path.join(RESULTS_DIR, f"{task}_rank{rank}", "results.json")
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            metric_key = TASKS[task]["metric"]
            score = data["metrics"].get(metric_key, None)
            scores.append(score)
        else:
            print(f"  Warning: missing results for {task} rank={rank} — skipping")
            scores.append(None)
    return scores
# ── Plot ───────────────────────────────────────────────────────────────────────
def plot_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "LoRA Rank Ablation — Performance vs. Rank r",
        fontsize=14,
        fontweight="normal",
        y=1.02,
    )
    for ax, (task, config) in zip(axes, TASKS.items()):
        scores = load_results(task)
        valid_ranks  = [r for r, s in zip(RANKS, scores) if s is not None]
        valid_scores = [s for s in scores if s is not None]
        ax.plot(
            valid_ranks,
            valid_scores,
            marker="o",
            linewidth=1.8,
            markersize=6,
            color=config["color"],
            label=f"{task.upper()} ({config['label']})",
        )
        for r, s in zip(valid_ranks, valid_scores):
            ax.annotate(
                f"{s:.3f}",
                xy=(r, s),
                xytext=(0, 8),
                textcoords="offset points",
                ha="center",
                fontsize=9,
                color="#444441",
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(RANKS)
        ax.set_xticklabels([str(r) for r in RANKS])
        ax.set_xlabel("LoRA Rank r", fontsize=11)
        ax.set_ylabel(config["label"], fontsize=11)
        ax.set_title(f"{task.upper()} — {config['label']} vs. Rank", fontsize=12)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", linewidth=0.5, alpha=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(fontsize=10)
    plt.tight_layout()
    out_path = os.path.join(RESULTS_DIR, "rank_ablation.png")
    os.makedirs(RESULTS_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nFigure saved to {out_path}")
    plt.show()
# ── Summary Table ──────────────────────────────────────────────────────────────
def print_summary():
    print("\n" + "=" * 55)
    print(f"  {'Rank':<8} {'SST-2 Accuracy':<20} {'MRPC F1':<15}")
    print("=" * 55)
    sst2_scores = load_results("sst2")
    mrpc_scores = load_results("mrpc")
    for rank, sst2, mrpc in zip(RANKS, sst2_scores, mrpc_scores):
        sst2_str = f"{sst2:.4f}" if sst2 is not None else "N/A"
        mrpc_str = f"{mrpc:.4f}" if mrpc is not None else "N/A"
        print(f"  r={rank:<6} {sst2_str:<20} {mrpc_str:<15}")
    print("=" * 55 + "\n")
# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("\nLoading results and generating figures...\n")
    print_summary()
    plot_ablation()
