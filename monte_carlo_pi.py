#!/usr/bin/env python3
"""
Monte Carlo Simulation to Estimate π

This script uses Monte Carlo simulation to estimate the value of π by:
1. Generating random points in a unit square
2. Counting points that fall inside the unit circle
3. Using the ratio to estimate π
4. Visualizing results with scatter plots and error analysis
"""

import argparse
import math
import random
import time
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from dataclasses import dataclass


@dataclass
class SimulationResult:
    """Container for simulation results"""
    sample_size: int
    pi_estimate: float
    error: float
    inside_points: int
    total_points: int
    execution_time: float


class MonteCarloPiEstimator:
    """Class to handle Monte Carlo simulation for π estimation"""
    
    def __init__(self, seed: Optional[int] = None):
        """Initialize the estimator with optional seed for reproducibility"""
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
    
    def generate_points(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate n random points in the unit square [0,1] x [0,1]"""
        x = np.random.uniform(0, 1, n)
        y = np.random.uniform(0, 1, n)
        return x, y
    
    def is_inside_circle(self, x: float, y: float) -> bool:
        """Check if point (x,y) is inside the unit circle"""
        return x**2 + y**2 <= 1
    
    def estimate_pi(self, n: int) -> SimulationResult:
        """Estimate π using n random points"""
        start_time = time.time()
        
        # Generate random points
        x, y = self.generate_points(n)
        
        # Count points inside the circle
        inside_mask = x**2 + y**2 <= 1
        inside_count = np.sum(inside_mask)
        
        # Calculate π estimate
        pi_estimate = 4 * inside_count / n
        
        # Calculate error
        error = abs(pi_estimate - math.pi)
        
        execution_time = time.time() - start_time
        
        return SimulationResult(
            sample_size=n,
            pi_estimate=pi_estimate,
            error=error,
            inside_points=inside_count,
            total_points=n,
            execution_time=execution_time
        )
    
    def run_multiple_simulations(self, sample_sizes: List[int]) -> List[SimulationResult]:
        """Run simulations for multiple sample sizes"""
        results = []
        for n in sample_sizes:
            result = self.estimate_pi(n)
            results.append(result)
        return results


class Visualizer:
    """Class to handle all visualization tasks"""
    
    def __init__(self):
        """Initialize the visualizer"""
        plt.style.use('default')
    
    def create_scatter_plot(self, x: np.ndarray, y: np.ndarray, 
                           inside_mask: np.ndarray, sample_size: int,
                           save_path: Optional[str] = None) -> None:
        """Create a scatter plot showing points inside vs outside the circle"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Plot points inside circle (blue)
        ax.scatter(x[inside_mask], y[inside_mask], 
                  c='blue', alpha=0.6, s=1, label='Inside circle')
        
        # Plot points outside circle (red)
        ax.scatter(x[~inside_mask], y[~inside_mask], 
                  c='red', alpha=0.6, s=1, label='Outside circle')
        
        # Draw the unit circle
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        
        # Draw the unit square
        square = plt.Rectangle((0, 0), 1, 1, fill=False, color='black', linewidth=2)
        ax.add_patch(square)
        
        # Calculate π estimate for this sample
        pi_estimate = 4 * np.sum(inside_mask) / len(inside_mask)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'Monte Carlo π Estimation\n'
                    f'Sample Size: {sample_size:,}\n'
                    f'π Estimate: {pi_estimate:.6f}\n'
                    f'True π: {math.pi:.6f}\n'
                    f'Error: {abs(pi_estimate - math.pi):.6f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_error_plot(self, results: List[SimulationResult], 
                         save_path: Optional[str] = None) -> None:
        """Create a plot showing error vs sample size"""
        sample_sizes = [r.sample_size for r in results]
        errors = [r.error for r in results]
        
        plt.figure(figsize=(12, 8))
        
        # Plot error vs sample size
        plt.subplot(2, 2, 1)
        plt.loglog(sample_sizes, errors, 'bo-', linewidth=2, markersize=6)
        plt.xlabel('Sample Size')
        plt.ylabel('Absolute Error')
        plt.title('Error vs Sample Size')
        plt.grid(True, alpha=0.3)
        
        # Plot π estimates vs sample size
        plt.subplot(2, 2, 2)
        pi_estimates = [r.pi_estimate for r in results]
        plt.semilogx(sample_sizes, pi_estimates, 'ro-', linewidth=2, markersize=6)
        plt.axhline(y=math.pi, color='black', linestyle='--', label=f'True π = {math.pi:.6f}')
        plt.xlabel('Sample Size')
        plt.ylabel('π Estimate')
        plt.title('π Estimate vs Sample Size')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot execution time vs sample size
        plt.subplot(2, 2, 3)
        execution_times = [r.execution_time for r in results]
        plt.loglog(sample_sizes, execution_times, 'go-', linewidth=2, markersize=6)
        plt.xlabel('Sample Size')
        plt.ylabel('Execution Time (seconds)')
        plt.title('Execution Time vs Sample Size')
        plt.grid(True, alpha=0.3)
        
        # Plot convergence (1/sqrt(n) vs error)
        plt.subplot(2, 2, 4)
        theoretical_error = [1 / math.sqrt(n) for n in sample_sizes]
        plt.loglog(theoretical_error, errors, 'mo-', linewidth=2, markersize=6)
        plt.xlabel('Theoretical Error Bound (1/√n)')
        plt.ylabel('Actual Error')
        plt.title('Convergence Analysis')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_animated_scatter(self, x: np.ndarray, y: np.ndarray, 
                               inside_mask: np.ndarray, sample_size: int,
                               frames: int = 100, interval: int = 50) -> None:
        """Create an animated scatter plot showing points being added gradually"""
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Draw the unit circle and square
        circle = plt.Circle((0, 0), 1, fill=False, color='black', linewidth=2)
        ax.add_patch(circle)
        square = plt.Rectangle((0, 0), 1, 1, fill=False, color='black', linewidth=2)
        ax.add_patch(square)
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_aspect('equal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.grid(True, alpha=0.3)
        
        # Initialize empty scatter plots
        inside_scatter = ax.scatter([], [], c='blue', alpha=0.6, s=1, label='Inside circle')
        outside_scatter = ax.scatter([], [], c='red', alpha=0.6, s=1, label='Outside circle')
        ax.legend()
        
        def animate(frame):
            # Calculate how many points to show in this frame
            points_per_frame = sample_size // frames
            current_points = min((frame + 1) * points_per_frame, sample_size)
            
            if current_points > 0:
                # Update inside points
                inside_x = x[:current_points][inside_mask[:current_points]]
                inside_y = y[:current_points][inside_mask[:current_points]]
                inside_scatter.set_offsets(np.column_stack([inside_x, inside_y]))
                
                # Update outside points
                outside_x = x[:current_points][~inside_mask[:current_points]]
                outside_y = y[:current_points][~inside_mask[:current_points]]
                outside_scatter.set_offsets(np.column_stack([outside_x, outside_y]))
                
                # Update title with current π estimate
                current_inside = np.sum(inside_mask[:current_points])
                current_pi_estimate = 4 * current_inside / current_points
                ax.set_title(f'Monte Carlo π Estimation (Animated)\n'
                           f'Points: {current_points:,}/{sample_size:,}\n'
                           f'π Estimate: {current_pi_estimate:.6f}')
            
            return inside_scatter, outside_scatter
        
        anim = animation.FuncAnimation(fig, animate, frames=frames, 
                                     interval=interval, blit=True, repeat=True)
        plt.show()
        return anim


def print_results(results: List[SimulationResult]) -> None:
    """Print simulation results in a formatted table"""
    print("\n" + "="*80)
    print("MONTE CARLO π ESTIMATION RESULTS")
    print("="*80)
    print(f"{'Sample Size':<12} {'π Estimate':<12} {'Error':<12} {'Time (s)':<10} {'Inside/Total':<15}")
    print("-"*80)
    
    for result in results:
        print(f"{result.sample_size:<12,} {result.pi_estimate:<12.6f} "
              f"{result.error:<12.6f} {result.execution_time:<10.4f} "
              f"{result.inside_points:,}/{result.total_points:,}")


def main():
    """Main function with CLI interface"""
    parser = argparse.ArgumentParser(
        description="Monte Carlo simulation to estimate π",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    

    
    args = parser.parse_args()
    
    # Initialize estimator and visualizer
    estimator = MonteCarloPiEstimator(seed=args.seed)
    visualizer = Visualizer()
    
    # Determine sample sizes to use
    if args.multiple:
        sample_sizes = args.multiple
    else:
        sample_sizes = [args.samples]
    
    print(f"Running Monte Carlo π estimation with sample sizes: {sample_sizes}")
    print(f"Random seed: {args.seed if args.seed else 'None (random)'}")
    
    # Run simulations
    results = estimator.run_multiple_simulations(sample_sizes)
    
    # Print results
    if not args.no_print:
        print_results(results)
    
    # Create visualizations
    if args.plot or args.animate:
        # Use the largest sample size for visualization
        largest_sample = max(sample_sizes)
        x, y = estimator.generate_points(largest_sample)
        inside_mask = x**2 + y**2 <= 1
        
        if args.plot:
            save_path = f"scatter_plot_{largest_sample}.png" if args.save else None
            visualizer.create_scatter_plot(x, y, inside_mask, largest_sample, save_path)
        
        if args.animate:
            visualizer.create_animated_scatter(x, y, inside_mask, largest_sample)
    
    if args.error_plot and len(results) > 1:
        save_path = "error_analysis.png" if args.save else None
        visualizer.create_error_plot(results, save_path)
    
    print(f"\nSimulation completed successfully!")


if __name__ == "__main__":
    main() 