#!/usr/bin/env python3
"""
SRAM Area and Power Estimator for SkyWater 130nm (Sky130)

This script implements a simple analytical model:
  - Area ≈ N_bits × A_bitcell × (1 + overhead)
  - Dynamic Power ≈ activity × W × C_bl × Vdd^2 × f
  - Leakage Power ≈ N_bits × I_leak_per_bit × Vdd

Defaults calibrated to a 6T bitcell in Sky130:
  • bitcell area = 1.896 μm²
  • overhead = 0.25 (25% for periphery & routing)
  • Vdd = 1.8 V
  • bitline capacitance = 200 fF per bit
  • switching activity = 0.1
  • leakage current per bit = 50 pA
"""

import argparse

def estimate_area(width, depth, bitcell_area=1.896, overhead=0.25):
    """
    Estimate total SRAM macro area in μm².
    
    width:   Word width in bits
    depth:   Number of words
    bitcell_area:  Area of one bitcell in μm²
    overhead:      Fractional overhead for decoders, sense amps, routing
    """
    n_bits = width * depth
    return n_bits * bitcell_area * (1 + overhead)

def estimate_power(width, depth, freq,
                   activity=0.1, C_bl=200e-15, Vdd=1.8, I_leak_per_bit=50e-12):
    """
    Estimate dynamic and leakage power in Watts.
    
    width:            Word width in bits
    depth:            Number of words
    freq:             Clock frequency in Hz
    activity:         Switching activity factor (0–1)
    C_bl:             Bitline capacitance per bit (F)
    Vdd:              Supply voltage (V)
    I_leak_per_bit:   Leakage current per bit (A)
    """
    # Dynamic power: assume one read per cycle across the full word
    P_dyn = activity * width * C_bl * Vdd**2 * freq
    
    # Leakage power: all bits leaking continuously
    n_bits = width * depth
    I_leak_total = I_leak_per_bit * n_bits
    P_leak = I_leak_total * Vdd
    
    return P_dyn, P_leak    

def main():
    parser = argparse.ArgumentParser(
        description="Estimate Sky130 SRAM macro area and power"
    )
    parser.add_argument("--width", type=int, required=True,
                        help="Word width (bits)")
    parser.add_argument("--depth", type=int, required=True,
                        help="Number of words")
    parser.add_argument("--clock_period", type=float, required=True,
                        help="Clock period in nanoseconds")
    parser.add_argument("--bitcell_area", type=float, default=1.896,
                        help="Bitcell area in μm² (default: 1.896)")
    parser.add_argument("--overhead", type=float, default=0.25,
                        help="Area overhead fraction (default: 0.25)")
    parser.add_argument("--activity", type=float, default=0.1,
                        help="Switching activity (default: 0.1)")
    parser.add_argument("--C_bl", type=float, default=200e-15,
                        help="Bitline capacitance per bit in F (default: 200e-15)")
    parser.add_argument("--Vdd", type=float, default=1.8,
                        help="Supply voltage in V (default: 1.8)")
    parser.add_argument("--I_leak", type=float, default=50e-12,
                        help="Leakage current per bit in A (default: 50e-12)")
    
    args = parser.parse_args()

    # convert clock period (ns) to frequency (Hz)
    freq_hz = 1.0 / (args.clock_period * 1e-9)
    freq_mhz = freq_hz / 1e6

    area_um2 = estimate_area(args.width, args.depth,
                             bitcell_area=args.bitcell_area,
                             overhead=args.overhead)
    P_dyn, P_leak = estimate_power(args.width, args.depth, freq_hz,
                                   activity=args.activity, C_bl=args.C_bl,
                                   Vdd=args.Vdd, I_leak_per_bit=args.I_leak)

    print(f"SRAM Configuration: {args.width} bits × {args.depth} words @ {args.clock_period:.3f} ns ({freq_mhz:.1f} MHz)")
    print(f"Estimated Area    : {area_um2:,.0f} μm² ({area_um2*1e-6:.3f} mm²)")
    print(f"Dynamic Power     : {P_dyn*1e3:.3f} mW")
    print(f"Leakage Power     : {P_leak*1e6:.3f} μW")
    print(f"Total Power       : {(P_dyn + P_leak)*1e3:.3f} mW")

if __name__ == "__main__":
    main()

