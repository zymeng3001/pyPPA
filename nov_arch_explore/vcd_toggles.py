#!/usr/bin/env python3
# vcd_toggles.py
# Summarize toggle counts for each signal in a VCD (no GUI needed)

from vcdvcd import VCDVCD
import math
import csv

VCD_PATH = "/home/yimengz/pyPPA/runs/core_top/1_core_top_sweep/1/objects/postsynth_sim/iverilog/core_top.vcd"
CSV_OUT  = None  # e.g. "toggles.csv" to dump a CSV

print(f"Reading {VCD_PATH} ...")
vcd = VCDVCD(VCD_PATH, store_tvs=True)  # add signals=[...] to limit scope if huge

def clean_val(v):
    # Normalize values; leave vectors as-is, but strip spaces
    return v.replace(" ", "") if isinstance(v, str) else v

toggle_stats = []  # (toggles, name, width, span_time, rate_per_timeunit)

# vcd.data: symbol -> Signal object with attributes: references (list), size (int), tv (list of (time,val))
for sym, sig in vcd.data.items():
    name = sig.references[0] if getattr(sig, "references", None) else sym
    tv = getattr(sig, "tv", None)
    if not tv or len(tv) < 2:
        continue

    toggles = 0
    last_val = clean_val(tv[0][1])
    for t, val in tv[1:]:
        val = clean_val(val)
        if val != last_val:
            toggles += 1
            last_val = val

    span = tv[-1][0] - tv[0][0]  # in VCD time units
    rate = (toggles / span) if span > 0 else math.nan
    width = getattr(sig, "size", 1)
    toggle_stats.append((toggles, name, width, span, rate))

toggle_stats.sort(key=lambda x: x[0], reverse=True)

print(f"Total signals with activity: {len(toggle_stats)}")
print("Top 20 by toggle count:")
for toggles, name, width, span, rate in toggle_stats[:20]:
    print(f"{toggles:8d}  {name}  (w={width})  span={span}  rate={rate:.6g}/timeunit")

if CSV_OUT:
    with open(CSV_OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["signal", "width", "toggles", "span_timeunits", "toggle_rate_per_timeunit"])
        for toggles, name, width, span, rate in toggle_stats:
            w.writerow([name, width, toggles, span, rate])
    print(f"Wrote {CSV_OUT}")
