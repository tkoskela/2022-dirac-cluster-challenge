#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numpy.random import default_rng
from time import perf_counter


SEED = 0


def set_dirichlet_bcs(x, bcs):
    """
    Apply Dirichlet boundary conditions; conditions that specify a particular value of x on the boundary.
    """
    left, top, right, bottom = bcs
    x[ 0, :] = left
    x[-1, :] = right
    x[ :,  0] = bottom
    x[ :, -1] = top


def set_u_bcs(u):
    u[ 0, :] = 0.
    u[-1, :] = 0.
    u[:,  0] = u[:,  2]
    u[:, -1] = u[:, -3]


def set_v_bcs(v):
    v[ 0, :] = v[ 2, :]
    v[-1, :] = v[-3, :]
    v[ :,  0] = 0.
    v[ :, -1] = 0.


def set_sfn_bcs(sfn):
    set_dirichlet_bcs(sfn, [0,0,0,0])


def set_vort_bcs(vort):
    set_dirichlet_bcs(vort, [0,0,0,0])


def set_tmp_bcs(tmp):
    tmp[ :,  0] = 1. # Bottom
    tmp[ :, -1] = 0 # Top

    tmp[ 0, :] = tmp[ 2, :]
    tmp[-1, :] = tmp[-3, :]


def calc_sfn(sfn, vort, dx, dy, max_iterations=200):
    """
    Calculates the stream function `sfn` from the equation $\nabla^2 \psi = - \omega$. This uses the Jacobi method to find the solution.
    """

    temp = np.zeros_like(sfn)
    sfn[:] = 0.0

    D = 2.*(1./dx**2 + 1./dy**2)

    for k in range(max_iterations):
        temp[1:-1,1:-1] = (vort[1:-1,1:-1] + (sfn[:-2,1:-1] + sfn[2:,1:-1])/dx**2 + (sfn[1:-1,:-2] + sfn[1:-1,2:])/dy**2)/D
        sfn[:] = temp[:]
        set_sfn_bcs(sfn)


def ddx(f, dx):
    """
    Returns x-derivative of f
    """
    return (f[2:,1:-1] - f[:-2,1:-1])/(2*dx)


def ddy(f, dy):
    """
    Returns y-derivative of f
    """
    return (f[1:-1:,2:] - f[1:-1,:-2])/(2*dy)


def nabla2(f, dx, dy):
    """
    Returns $\nabla^2 f$, i.e. the Laplacian, i.e. $d^2f/dx^2 + d^2f/dy^2$
    """
    return (f[:-2,1:-1] - 2*f[1:-1,1:-1] + f[2:,1:-1])/dx**2 + (f[1:-1,:-2] - 2*f[1:-1,1:-1] + f[1:-1,2:])/dy**2


def update(f, dfdt, dfdt_prev, dt):
    """
    Return f advanced by one timestep using its derivative and previous value (this is called the Adams-Bashforth method)
    """
    return f + dt/2.*(3.*dfdt - dfdt_prev)


def main():
    # Parameters
    nx = 64
    ny = 64

    lx = 1.
    ly = 1.
    Pr = 10 # CHANGEME to get different physics (Prandtl number)
    Ra = 1e6 # CHANGEME to get different physics (Rayleigh number)

    dx = lx/nx
    dy = ly/ny

    dt = 0.5*min(dx,dy)**2/(4*Pr) # CHANGEME if you change nx or ny and the simulation gets freaky (ask Jamie)
    total_time = 0.01 # CHANGEME to change length of entire simulation
    dump_dt = total_time*0.1 # CHANGEME to get more/fewer plots (for tasty animations)
    print_dt = total_time/20

    FP_TYPE = np.float64 # CHANGEME to change datatype

    print(f'dt: {dt}')
    print(f'dx: {dx}')
    print(f'Ra: {Ra}')

    # Setup variables
    u = np.zeros((nx+2, ny+2), dtype=FP_TYPE)
    v = np.zeros_like(u)

    sfn = np.zeros_like(u)
    vort = np.zeros_like(u)
    dvortdt = np.zeros_like(u)
    dvortdt_prev = np.zeros_like(u)

    tmp = np.zeros_like(u)
    dtmpdt = np.zeros_like(u)
    dtmpdt_prev = np.zeros_like(u)

    rng = default_rng(SEED)

    x = np.linspace(-dx, lx+dx, nx+2)
    y = np.linspace(-dy, ly+dy, ny+2)
    X, Y = np.meshgrid(x, y, indexing='ij')

    # Set initial conditions
    tmp = 1. - Y/ly
    epsilon = 1e-2
    tmp += epsilon * (2 * rng.random(tmp.shape) - 1.0)
    # tmp += epsilon * np.sin(np.pi*X)
    # vort += epsilon * np.sin(np.pi*X)

    fig, ax = plt.subplots()

    # Start timer
    start = perf_counter()

    t = 0.
    step_counter = 0
    time_to_next_dump = 0.
    time_to_next_print = 0.
    dump_counter = 0
    while t < total_time:
        if (t >= time_to_next_dump):
            plot = plt.imshow(tmp.T, origin='lower', extent=(0., lx, 0., ly))
            plt.colorbar(plot)
            if dump_counter != 0:
                plt.quiver(x, y, u.T, v.T)

            plt.xlim(0, lx)
            plt.ylim(0, ly)

            fname = f"{dump_counter:04d}.png"
            print(f"Saving {fname}")
            plt.savefig(fname)
            plt.close()

            time_to_next_dump += dump_dt
            dump_counter += 1

        if (t >= time_to_next_print):
            print(f"t={t:.4f}/{total_time} ({t/total_time*100:.2f}%)")
            time_to_next_print += print_dt

        # Figure out stream function
        calc_sfn(sfn, vort, dx, dy)

        # Calculate actual velocities
        u[1:-1,1:-1] = -ddy(sfn, dx)
        v[1:-1,1:-1] =  ddx(sfn, dy)

        set_u_bcs(u)
        set_v_bcs(v)

        # Calculate time derivatives
        dvortdt[1:-1,1:-1] = \
                -(u[1:-1,1:-1] * ddx(vort, dx) + v[1:-1,1:-1] * ddy(vort, dy))\
                + Pr*nabla2(vort, dx, dy)\
                - Ra*Pr*ddx(tmp, dx)

        dtmpdt[1:-1,1:-1] = \
                -(u[1:-1,1:-1] * ddx(tmp, dx) + v[1:-1,1:-1] * ddy(tmp, dy))\
                + nabla2(tmp, dx, dy)

        # Update variables
        vort[:] = update(vort, dvortdt, dvortdt_prev, dt)
        tmp[:] = update(tmp, dtmpdt, dtmpdt_prev, dt)

        # Set boundary conditions
        set_vort_bcs(vort)
        set_tmp_bcs(tmp)

        dvortdt_prev[:] = dvortdt
        dtmpdt_prev[:] = dtmpdt

        t += dt
        step_counter += 1

    end = perf_counter()
    print(f"Time to complete: {end-start}")


if __name__ == "__main__":
    main()
