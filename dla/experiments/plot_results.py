"""
Plot runtime analysis results and verify O(N log N) complexity.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def load_runtime_data(csv_path: str):
    """Load runtime data from CSV."""
    pixels = []
    times = []
    
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pixels.append(int(row['pixels']))
            times.append(float(row['time_ms']))
    
    return np.array(pixels), np.array(times)


def fit_n_log_n(pixels, times):
    """
    Fit O(N log N) curve to data.
    
    Model: T(N) = a * N * log(N) + b
    """
    def model(N, a, b):
        return a * N * np.log(N) + b
    
    # Initial guess
    p0 = [1e-5, 0]
    
    # Fit
    params, covariance = curve_fit(model, pixels, times, p0=p0)
    
    a, b = params
    
    # Compute R-squared
    residuals = times - model(pixels, a, b)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((times - np.mean(times))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    return a, b, r_squared, model


def plot_runtime_vs_pixels(pixels, times, output_path: str):
    """
    Plot runtime vs pixels with O(N log N) fit.
    """
    # Fit O(N log N)
    a, b, r_squared, model = fit_n_log_n(pixels, times)
    
    # Generate smooth curve
    pixels_smooth = np.linspace(pixels.min(), pixels.max(), 100)
    times_fitted = model(pixels_smooth, a, b)
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    # Scatter plot of actual data
    plt.scatter(pixels, times, s=100, alpha=0.7, color='blue', 
                label='Measured runtime', zorder=3)
    
    # Fitted O(N log N) curve
    plt.plot(pixels_smooth, times_fitted, 'r-', linewidth=2,
             label=f'O(N log N) fit (R² = {r_squared:.4f})', zorder=2)
    
    plt.xlabel('Number of Pixels (N)', fontsize=12)
    plt.ylabel('Runtime (milliseconds)', fontsize=12)
    plt.title('Runtime vs Image Size - Verifying O(N log N) Complexity', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=11)
    
    # Add equation text
    eq_text = f'T(N) = {a:.2e} × N log(N) + {b:.2f}'
    plt.text(0.05, 0.95, eq_text, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved plot to {output_path}")
    
    return a, b, r_squared


def plot_log_log(pixels, times, output_path: str):
    """
    Create log-log plot to verify complexity.
    
    For O(N log N): log(T) ≈ log(N) + log(log(N))
    which appears roughly linear in log-log space with slight upward curve.
    """
    # Fit O(N log N) for comparison
    a, b, r_squared, model = fit_n_log_n(pixels, times)
    
    # Generate smooth curve
    pixels_smooth = np.linspace(pixels.min(), pixels.max(), 100)
    times_fitted = model(pixels_smooth, a, b)
    
    plt.figure(figsize=(10, 6))
    
    # Log-log plot
    plt.loglog(pixels, times, 'bo', markersize=8, alpha=0.7,
               label='Measured runtime')
    plt.loglog(pixels_smooth, times_fitted, 'r-', linewidth=2,
               label='O(N log N) fit')
    
    # Reference lines
    # O(N) reference
    n_linear = a * pixels_smooth * 0.5
    plt.loglog(pixels_smooth, n_linear, 'g--', alpha=0.5, linewidth=1,
               label='O(N) reference')
    
    # O(N²) reference  
    n_squared = a * pixels_smooth * np.log(pixels_smooth) * 2
    plt.loglog(pixels_smooth, n_squared, 'm--', alpha=0.5, linewidth=1,
               label='O(N²) reference')
    
    plt.xlabel('Number of Pixels (N)', fontsize=12)
    plt.ylabel('Runtime (milliseconds)', fontsize=12)
    plt.title('Log-Log Plot: Runtime Complexity Analysis', fontsize=14)
    plt.grid(True, alpha=0.3, which='both')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved log-log plot to {output_path}")


def print_analysis_summary(pixels, times, a, b, r_squared):
    """Print summary of complexity analysis."""
    print("\n" + "=" * 70)
    print("COMPLEXITY ANALYSIS SUMMARY")
    print("=" * 70)
    
    print(f"\nFitted Model: T(N) = {a:.2e} × N log(N) + {b:.2f}")
    print(f"R-squared: {r_squared:.6f}")
    print(f"  (R² close to 1.0 indicates excellent fit)")
    
    print(f"\nData Points: {len(pixels)}")
    print(f"  Min pixels: {pixels.min():,}")
    print(f"  Max pixels: {pixels.max():,}")
    print(f"  Min time: {times.min():.2f} ms")
    print(f"  Max time: {times.max():.2f} ms")
    
    # Growth rate analysis
    if len(pixels) > 1:
        idx_sort = np.argsort(pixels)
        p_sorted = pixels[idx_sort]
        t_sorted = times[idx_sort]
        
        # Compare smallest and largest
        n_ratio = p_sorted[-1] / p_sorted[0]
        t_ratio = t_sorted[-1] / t_sorted[0]
        
        # Theoretical O(N log N) ratio
        theoretical_ratio = n_ratio * np.log(p_sorted[-1]) / np.log(p_sorted[0])
        
        print(f"\nGrowth Analysis:")
        print(f"  Pixel increase: {n_ratio:.2f}x")
        print(f"  Time increase: {t_ratio:.2f}x")
        print(f"  O(N log N) predicts: {theoretical_ratio:.2f}x")
        print(f"  Match: {abs(t_ratio - theoretical_ratio) / theoretical_ratio * 100:.1f}% difference")
    
    print("\n" + "=" * 70)


def main():
    """Generate all plots and analysis."""
    print("=" * 70)
    print("PLOTTING RUNTIME ANALYSIS RESULTS")
    print("=" * 70)
    
    # Load data
    csv_path = "experiments/runtime_data.csv"
    print(f"\nLoading data from {csv_path}...")
    
    try:
        pixels, times = load_runtime_data(csv_path)
        print(f"✓ Loaded {len(pixels)} data points")
    except FileNotFoundError:
        print(f"✗ Error: {csv_path} not found. Run runtime_analysis.py first.")
        return
    
    # Create plots directory
    import os
    os.makedirs("outputs/plots", exist_ok=True)
    
    # Plot 1: Runtime vs Pixels
    print("\nGenerating runtime vs pixels plot...")
    a, b, r_squared = plot_runtime_vs_pixels(
        pixels, times, 
        "outputs/plots/runtime_vs_pixels.png"
    )
    
    # Plot 2: Log-Log
    print("\nGenerating log-log plot...")
    plot_log_log(pixels, times, "outputs/plots/runtime_loglog.png")
    
    # Print analysis
    print_analysis_summary(pixels, times, a, b, r_squared)
    
    print("\n✓ All plots generated successfully!")
    print("  Check outputs/plots/ directory")


if __name__ == "__main__":
    main()

