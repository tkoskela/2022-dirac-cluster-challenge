## DiRAC Cluster Challenge 2022-05-26

Welcome to the starting codes for the cluster challenge! There are two challenge codes here:

### `nbody.py`

This little code simulates the gravitational interaction between many astronomical bodies. In fact, this kind of algorithm is quite general; as well as simulating gravitational phenomena like galaxy formation, Saturn's rings and planetary systems, it can be used to simulate fluids, electrodynamics and even some statistical processes. This code includes two examples, a version of our solar system and a completely randomised solar system with as many planets (or bodies) as you like.

The main algorithm has two steps:

1. Calculate the gravitational force on one body from all other bodies (and do this for every single body)
2. Move all bodies forward in time

The computationally intensive part of this is step 1. If there are $N$ bodies, the code must calculate the force on one body from the $N-1$ other bodies, i.e. $N-1$ calculations, and it must repeat that for every body, so $N \times (N-1) \approx N^2$ calculations. For the planets in the solar system this is fine but if we wanted to simulate all the astroids as well (over a million!) this algorithm becomes expensive very quickly. There are two solutions; we can choose a more complex algorithm, or we can speed this up with HPC! Since this is *not* a workshop on numerical methods, we'll be speeding this up. Luckily the algorithm suits parallelisation quite well.

### `rayleigh_bendard.py`

This code simulates a phenomenon called [Rayleigh-Bénard convection](https://www.youtube.com/watch?v=OM0l2YPVMf8), where fluid in a box is heated from below and (since heat rises) we get convection! This is a slightly trickier code to understand but I'll explain the broad overview:

- We'll simulate the fluid in a square box heated from below
- The fluid is represented by three variables:
  - temperature which measures the temperature at each point in the domain
  - vorticity which measures the "swirliness" of the fluid at every point
  - a "stream function" which describes the velocity of the fluid (but allows us to ignore the velocity)
- To simulate the evolution of these three variables we have three equations which I'll describe in words:
  - The stream function is calculated from the vorticity and then differentiated to get the x and y velocity components
  - the temperature is moved by the velocity and it diffuses through the fluid
  - the vorticity is also moved by the velocity, it also diffuses and, crucially, we include a term which generates vorticity from differences in temperature
- In order to solve these three equations we use:
  - The Jacobi method to solve for the stream function (this solves a matrix equation)
  - Finite difference and Adams-Bashforth methods to evolve the vorticity and temperature equations in time. This is all more obvious from the code 
