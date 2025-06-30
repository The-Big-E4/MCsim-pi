# Monte Carlo Estimation of Pi

This project uses a Monte Carlo simulation to estimate the value of π by generating random points inside a unit square and calculating the ratio of points that fall inside a quarter circle.

## Files

- **MCmain.py** – A simple script to run simulations with multiple sample sizes and visualize results. Intended for demonstration use.
- **monte_carlo_pi.py** – Core module containing classes for simulation, visualization, and optional CLI support.

## How It Works

1. Generate random (x, y) points in the unit square [0, 1] × [0, 1].
2. Count the number of points that fall inside the quarter circle (x² + y² ≤ 1).
3. Use the ratio π ≈ 4 × (points inside circle) / (total points) to estimate π.
4. Visualize results via scatter plots and error analysis.

## Running a Demo

To run a demo with predefined sample sizes and visualizations:

```bash
python MCmain.py
```

This script runs simulations for 10⁴, 10⁵, and 10⁶ samples.  
You can adjust the sample sizes in `MCmain.py` (line 15), but be aware that extremely large numbers may crash the program.

The script prints out the estimated π values, errors, and execution times, and generates:

- A scatter plot for the largest sample size
- A convergence/error analysis plot

## Dependencies

- Python 3.7+
- NumPy
- Matplotlib

Install dependencies with:

```bash
pip install numpy matplotlib
```

## Output Example

```
Sample Size   π Estimate    Error        Time (s)   Inside/Total
---------------------------------------------------------------
10,000        3.141200      0.000392     0.0152     7,854/10,000
```

## Key Concepts

- Monte Carlo estimation
- Convergence analysis
- Probabilistic modeling
- Error bounds vs. theoretical 1/√n decay

## License

MIT License  
You may use this code for educational or reference purposes, but please do not modify or re-upload altered versions without permission.
