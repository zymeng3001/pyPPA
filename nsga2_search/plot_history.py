#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Dict, Any
import pandas as pd
import matplotlib.pyplot as plt


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")
    recs = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                recs.append(json.loads(line))
            except json.JSONDecodeError:
                # skip malformed lines
                continue
    if not recs:
        raise RuntimeError(f"No records found in {path}")
    return recs


def to_dataframe(recs: List[Dict[str, Any]]) -> pd.DataFrame:
    # Expected fields: ts, gen, pareto_size, avg_objs (tuple of 4 floats)
    rows = []
    for r in recs:
        row = {
            'ts': r.get('ts'),
            'gen': r.get('gen'),
            'pareto_size': r.get('pareto_size'),
        }
        avg = r.get('avg_objs') or []
        # pad/trim to 4 objectives for stable plotting
        while len(avg) < 4:
            avg.append(None)
        if len(avg) > 4:
            avg = avg[:4]
        row.update({
            'avg_val_loss': avg[0],
            'avg_neg_throughput': avg[1],
            'avg_energy_token': avg[2],
            'avg_ttft_ms': avg[3],
        })
        rows.append(row)
    df = pd.DataFrame(rows).sort_values('gen')
    return df


def plot_history(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)

    # Pareto size over generations
    plt.figure(figsize=(7,4))
    plt.plot(df['gen'], df['pareto_size'], marker='o', lw=1.5)
    plt.xlabel('Generation')
    plt.ylabel('Pareto front size')
    plt.title('NSGA-II Pareto Size vs Generation')
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    p1 = os.path.join(out_dir, 'pareto_size_vs_gen.png')
    plt.savefig(p1, dpi=200)

    # Avg objectives over generations
    fig, ax = plt.subplots(2, 2, figsize=(10,6), sharex=True)
    ax = ax.flatten()
    series = [
        ('avg_val_loss', 'Avg Val Loss (min)'),
        ('avg_neg_throughput', 'Avg -Throughput (min)'),
        ('avg_energy_token', 'Avg Energy/Token (min)'),
        ('avg_ttft_ms', 'Avg TTFT (ms, min)'),
    ]
    for i, (col, label) in enumerate(series):
        ax[i].plot(df['gen'], df[col], marker='o', lw=1.2)
        ax[i].set_ylabel(label)
        ax[i].grid(True, alpha=0.25)
    ax[-1].set_xlabel('Generation')
    fig.suptitle('NSGA-II Average Objective Summary (Pareto front)')
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    p2 = os.path.join(out_dir, 'avg_objectives_vs_gen.png')
    plt.savefig(p2, dpi=200)

    print(f"Wrote plots:\n - {p1}\n - {p2}")


def main():
    parser = argparse.ArgumentParser(description='Plot NSGA-II history from JSONL logs.')
    parser.add_argument('--log', default='./data/hetero_nsga2_run.jsonl', help='Path to JSONL log')
    parser.add_argument('--out', default='./plots', help='Output directory for plots')
    args = parser.parse_args()

    recs = load_jsonl(args.log)
    df = to_dataframe(recs)
    plot_history(df, args.out)


if __name__ == '__main__':
    main()
