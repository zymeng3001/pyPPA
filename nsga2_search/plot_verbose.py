#!/usr/bin/env python3
import argparse
import json
import os
from typing import Any, Dict, List, Union
import matplotlib.pyplot as plt

try:
    import imageio.v2 as imageio
except Exception:
    imageio = None


def load_verbose_records(path: str) -> List[Dict[str, Any]]:
    """Load verbose NSGA-II log records.

    Supports two formats:
      1. JSONL (one JSON object per line) produced by make_jsonl_logger (preferred for verbose logging)
      2. Pretty-printed JSON artifact (full object) containing a 'history' list (non-verbose summary only)

    Returns a list of dict records. Non-dict JSON lines/items are ignored.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Log file not found: {path}")

    # First try to parse entire file as a single JSON value (artifact style or list)
    try:
        with open(path, 'r') as f:
            whole = json.load(f)
        # If it's a list of dicts, use directly
        if isinstance(whole, list) and all(isinstance(x, dict) for x in whole):
            return whole
        # If it's a dict with 'history', expand history entries (no population/offspring info)
        if isinstance(whole, dict) and 'history' in whole:
            hist = whole['history'] or []
            if isinstance(hist, list):
                # ensure each is a dict with gen
                conv = []
                for h in hist:
                    if isinstance(h, dict):
                        conv.append(h)
                if conv:
                    print("Info: artifact lacks per-individual verbose data; only history summaries available.")
                    return conv
        # If it's some other JSON type, fall back to JSONL line parsing below
    except json.JSONDecodeError:
        pass  # not a single JSON structure; treat as JSONL
    except Exception:
        # unexpected error, continue to line parsing
        pass

    # Fallback: treat as JSONL
    recs: List[Dict[str, Any]] = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                recs.append(obj)
    if not recs:
        raise RuntimeError(f"No usable records found in {path}. If you want verbose plots, run nsga2 with verbose=True and log_fn set.")
    return recs


def plot_gen(state: Dict[str, Any], out_dir: str, x_idx: int = 0, y_idx: int = 1) -> str:
    gen = state.get('gen')
    pop = state.get('population')
    off = state.get('offspring')

    if pop is None or off is None:
        # Not a verbose record
        return ""

    plt.figure(figsize=(7,5))
    # Separate Pareto vs non-Pareto in population if flag available
    pareto_mask = [p.get('on_pareto', False) for p in pop]
    xs_pop = [p['objs'][x_idx] for p in pop]
    ys_pop = [p['objs'][y_idx] for p in pop]
    xs_pf = [x for x, m in zip(xs_pop, pareto_mask) if m]
    ys_pf = [y for y, m in zip(ys_pop, pareto_mask) if m]
    xs_np = [x for x, m in zip(xs_pop, pareto_mask) if not m]
    ys_np = [y for y, m in zip(ys_pop, pareto_mask) if not m]
    if xs_np:
        plt.scatter(xs_np, ys_np, c='tab:blue', s=22, label='Population', alpha=0.55)
    if xs_pf:
        plt.scatter(xs_pf, ys_pf, c='tab:green', s=38, marker='*', label='Pareto Front', edgecolors='black', linewidths=0.4)

    # Plot offspring
    xs_sel = [o['objs'][x_idx] for o in off if o.get('selected')]
    ys_sel = [o['objs'][y_idx] for o in off if o.get('selected')]
    xs_nsel = [o['objs'][x_idx] for o in off if not o.get('selected')]
    ys_nsel = [o['objs'][y_idx] for o in off if not o.get('selected')]
    if xs_nsel:
        plt.scatter(xs_nsel, ys_nsel, c='tab:gray', s=18, label='Offspring (not selected)', alpha=0.6)
    if xs_sel:
        plt.scatter(xs_sel, ys_sel, c='tab:orange', s=26, marker='^', label='Offspring (selected)', alpha=0.9)

    plt.xlabel(f'Objective {x_idx} (minimize)')
    plt.ylabel(f'Objective {y_idx} (minimize)')
    pop_sz = state.get('population_size', len(pop))
    off_gen = state.get('offspring_generated', len(off))
    off_sel = state.get('offspring_selected', sum(1 for o in off if o.get('selected')))
    pf_count = sum(pareto_mask)
    plt.title(f'Gen {gen}: pop={pop_sz} pf={pf_count} off(gen={off_gen}, sel={off_sel})')
    plt.grid(True, alpha=0.25)
    plt.legend(loc='best')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'verbose_gen_{gen:03d}.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=180)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description='Visualize verbose NSGA-II logs (population & offspring).')
    parser.add_argument('--log', default='./data/hetero_nsga2_run.jsonl', help='Path to JSONL log')
    parser.add_argument('--out', default='./plots', help='Output directory for images')
    parser.add_argument('--x', type=int, default=0, help='Objective index for X axis')
    parser.add_argument('--y', type=int, default=1, help='Objective index for Y axis')
    parser.add_argument('--gif', action='store_true', help='Also build an animated GIF (requires imageio)')
    args = parser.parse_args()

    recs = load_verbose_records(args.log)
    out_imgs = []
    for st in recs:
        if not isinstance(st, dict):
            continue
        p = plot_gen(st, args.out, x_idx=args.x, y_idx=args.y)
        if p:
            out_imgs.append(p)

    if out_imgs:
        print(f"Wrote {len(out_imgs)} verbose generation images to {args.out}")
    else:
        print("No verbose population/offspring entries found to plot (did you run with verbose=True?).")

    if args.gif and out_imgs:
        if imageio is None:
            print('imageio not available; skipping GIF creation')
        else:
            gif_path = os.path.join(args.out, 'verbose_evolution.gif')
            frames = [imageio.imread(p) for p in out_imgs]
            imageio.mimsave(gif_path, frames, duration=0.9)
            print(f"Wrote GIF: {gif_path}")


if __name__ == '__main__':
    main()
