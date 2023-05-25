import numpy as np
from matplotlib import pyplot as plt
import sys

# basic solver to compute the time dependent magnetic diffusion equation in 2D, with fixed boundary conditions

# input parameters
n = 101
L = 10.0
h = L/(n-1)
eta = 1.22
total_time = 5
dt = (1/n/n)*(1/2/eta)
n_steps = int(total_time/dt)
eps = 1e-3

# wavelengths for initial field
Lxx = 1.2
Lxy = 0.1
Lyx = 0.2
Lyy = 0.6

# boundary conditions
Bx_top = 1
Bx_bottom = 1
Bx_left = 1
Bx_right = 1
By_top = 1
By_bottom = 1
By_left = 1
By_right = 1

# define functions:
# initialise magnetic field component
def init_B(xgrid, ygrid, B_top, B_bottom, B_left, B_right, Lx, Ly):
    B = np.sin(Lx*xgrid)*np.cos(Ly*ygrid)
    B[-1,:] = B_top
    B[0,:] = B_bottom
    B[:,0] = B_left
    B[:,-1] = B_right
    return B

# form laplacian 
def form_laplacian(f,h):
    return (f[2:,1:-1] + f[0:-2,1:-1] + f[1:-1,:-2] + f[1:-1,2:] - 4.0*f[1:-1,1:-1])/(h**2.0)

# function to solve component of mag field
def solve_component(B, total_time, h, eta, dt, L, eps):
    residual = 10
    time = 0
    # iterate over time
    while(time < total_time):

        # calculate laplacian term
        delta_B = form_laplacian(B,h)

        # original for residual
        B_prev = np.copy(B)

        # get centre to update
        B_c = np.copy(B[1:-1,1:-1])
        B[1:-1,1:-1] = B_c + eta*dt/L/L*delta_B

        # calculate a residual
        residual = np.sum(np.abs((B - B_prev)/B))

        print(f"t={time}, r={residual}")
        if(residual < eps):
            print("solution converged")
            break
        # advance
        time += dt
    return B

# set up initial grid
xgrid, ygrid = np.meshgrid(np.linspace(0,L,n), np.linspace(0,L,n))

# initialise field
Bx = init_B(xgrid,ygrid, Bx_top, Bx_bottom, Bx_left, Bx_right,Lxx,Lxy)
By = init_B(xgrid,ygrid, By_top, By_bottom, By_left, By_right,Lyx,Lyy)

# calculate magnitude
mag_B_orig = np.sqrt(Bx*Bx + By*By)

# save original for plotting
Bx_orig = np.copy(Bx)
By_orig = np.copy(By)

# solve each component
Bx = solve_component(Bx,total_time,h,eta,dt,L,eps)
By = solve_component(By,total_time,h,eta,dt,L,eps)
# calculate magnitude
mag_B = np.sqrt(Bx*Bx + By*By)

# plot
fig,(ax_orig, ax) = plt.subplots(1,2, sharey=True)
ax_orig.streamplot(xgrid,ygrid, Bx_orig,By_orig,color=mag_B_orig, density=1.4, linewidth=None)
ax.streamplot(xgrid,ygrid, Bx,By, color=mag_B, density=1.4, linewidth=None)
ax_orig.set(xlabel="x", title="original")
ax.set(xlabel="x", ylabel="y", title="final")
plt.tight_layout()
plt.show()

