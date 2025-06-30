Monte Carlo Estimation of Pi
This project uses a Monte Carlo simulation to estimate the value of π by generating random points inside a unit square and calculating the ratio of points that fall inside a quarter circle.

Files
  MCmain.py: A simple script to run simulations with multiple sample sizes and visualize results. Intended for demonstration use.
  monte_carlo_pi.py: Core module containing classes for simulation, visualization, and optional CLI support.

How It Works
  Generate random (x, y) points in the unit square [0, 1] × [0, 1].
  Count the number of points that fall inside the quarter circle (x² + y² ≤ 1).
  Use the ratio π ≈ 4 × (points inside circle) / (total points) to estimate π.
  Visualize results via scatter plots and error analysis.

Running A Demo
  To run a demo with predefined sample sizes and visualizations:
  "python MCmain.py"
  This script runs simulations for 10^4, 10^5, and 10^6, but you can adjust the 0's in MCmain.py line 15, but beware too large of a number can crash the program. The script prints out the estimated π values, errors, and execution times, and generates:
  A scatter plot for the largest sample size
  A convergence/error analysis plot

Dependencies
  Python 3.7+
  NumPy
  Matplotlib

Output
  The script prints formatted results such as:

  Sample Size   π Estimate    Error        Time (s)   Inside/Total
  -----------------------------------------------------------------
  10,000        3.141200      0.000392     0.0152     7,854/10,000
  -----------------------------------------------------------------

Key Concepts
  Monte Carlo estimation
  Convergence analysis
  Probabilistic modeling
  Error bounds vs. theoretical 1/√n decay

MIT Liscense
You may use this code for educational or reference purposes, but please do not modify or re-upload altered versions without permission.
