#Main, Reccomend not 
from monte_carlo_pi import MonteCarloPiEstimator, Visualizer, print_results

def main():
    #Run a simple demonstration of the Monte Carlo π estimation
    
    print("Monte Carlo π Estimation - Simple Example")
    print("=" * 50)
    
    # Initialize the estimator with a fixed seed for reproducible results
    estimator = MonteCarloPiEstimator(seed=42)
    visualizer = Visualizer()
    
    # Define sample sizes to test
    sample_sizes = [10000, 100000, 1000000]
    
    print(f"Running simulations with sample sizes: {sample_sizes}")
    
    # Run simulations
    results = estimator.run_multiple_simulations(sample_sizes)
    
    # Print results
    print_results(results)
    
    # Create a scatter plot for the largest sample size
    largest_sample = max(sample_sizes)
    print(f"\nGenerating scatter plot for {largest_sample:,} points...")
    
    x, y = estimator.generate_points(largest_sample)
    inside_mask = x**2 + y**2 <= 1
    
    # Create and show the scatter plot
    visualizer.create_scatter_plot(x, y, inside_mask, largest_sample)
    
    # Create error analysis plots
    visualizer.create_error_plot(results)
    
    print("Simulatoin completed. Check the generated plots.")

if __name__ == "__main__":
    main() 