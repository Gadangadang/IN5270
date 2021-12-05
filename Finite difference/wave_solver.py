import numpy as np
import matplotlib.pyplot as plt
import time
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import matplotlib.animation as animation
import seaborn as sns
from tqdm import trange


class waveSolver:
    def __init__(self, dx, dy, Lx, Ly, T, dt, I, V, f, q, b):
        self.T = T
        self.q = q
        self.I = I
        #self.c = c
        self.b = b
        self.n = 0

        self.dx = dx
        self.dy = dy
        self.dt = dt

        self.Nt = int(self.T / self.dt + 1)
        self.Nx = int(Lx / self.dx + 1)
        self.Ny = int(Ly / self.dy + 1)

        self.x = np.linspace(0, Lx, self.Nx)  # mesh points in x dir
        self.y = np.linspace(0, Ly, self.Ny)  # mesh points in y dir
        self.t = np.linspace(0, self.T, self.Nt)  # mesh points in time

        self.xv, self.yv = np.meshgrid(self.x, self.y)

        # Allow f and V to be None or 0
        if f is None or f == 0:
            def f(x, y, t):
                return np.zeros((x.shape[0], y.shape[1]))
        else:
            self.f = f

            # or simpler: x*y*0
        if V is None or V == 0:

            def V(x, y):
                return np.zeros((x.shape[0], y.shape[1]))
        else:
            self.V = V

        self.u = np.zeros((self.Nx + 2, self.Ny + 2))  # solution array
        self.u1 = np.zeros((self.Nx + 2, self.Ny + 2))  # solution at t-dt
        self.u2 = np.zeros((self.Nx + 2, self.Ny + 2))  # solution at t-2*dt

        self.u[1:-1, 1:-1] = self.I(self.xv, self.yv)
        # correct for ghost points
        self.u = self.ghost_correction(self.u)

        self.create_q()

    def create_q(self):
        self.q_half_minus = np.array(
            [np.zeros((self.Nx + 2, self.Ny + 2)), np.zeros((self.Nx + 2, self.Ny + 2))]
        )
        self.q_half_plus = np.array(
            [np.zeros((self.Nx + 2, self.Ny + 2)), np.zeros((self.Nx + 2, self.Ny + 2))]
        )
        self.q_vals = np.zeros((self.Nx + 2, self.Ny + 2))

        self.q_vals[1:-1, 1:-1] = self.q(self.xv, self.yv)

        self.q_half_plus[0, 1:-1, 1:-1] = 0.5 * (
            self.q_vals[2:, 1:-1] + self.q_vals[:-2, 1:-1]
        )
        self.q_half_minus[0, 1:-1, 1:-1] = 0.5 * (
            self.q_vals[1:-1, 1:-1] + self.q_vals[:-2, 1:-1]
        )
        self.q_half_plus[1, 1:-1, 1:-1] = 0.5 * (
            self.q_vals[1:-1, 2:] + self.q_vals[1:-1, :-2]
        )
        self.q_half_minus[1, 1:-1, 1:-1] = 0.5 * (
            self.q_vals[1:-1, 1:-1] + self.q_vals[1:-1, :-2]
        )

    def advance(self):

        if self.n == 0:  # Modified scheme for n=0
            self.u1 = self.u.copy()  # move over solution array index
            self.u[1:-1, 1:-1] = (
                2 * self.I(self.xv, self.yv)
                + (1 - self.b * self.dt / 2) * (2 * self.dt * self.V(self.xv, self.yv))
                + self.dt ** 2 * self.space()
            ) / 2
            self.u = self.ghost_correction(self.u)
            self.n += 1  # update timestep

        else:  # General scheme
            # move over solution array index
            self.u2, self.u1, self.u = self.u1, self.u, self.u2
            self.u[1:-1, 1:-1] = (
                2 * self.u1[1:-1, 1:-1]
                + (self.b * self.dt / 2 - 1) * self.u2[1:-1, 1:-1]
                + self.dt ** 2 * self.space()
            ) / (1 + self.b * self.dt / 2)
            self.u = self.ghost_correction(self.u)
            self.n += 1  # update timestep

    def space(self):

        ddx = (
            1
            / self.dx ** 2
            * (
                self.q_half_plus[0, 1:-1, 1:-1]
                * (self.u1[1:, 1:-1] - self.u1[:-1, 1:-1])[1:, :]
                - self.q_half_minus[0, 1:-1, 1:-1]
                * (self.u1[1:, 1:-1] - self.u1[:-1, 1:-1])[:-1, :]
            )
        )
        ddy = (
            1
            / self.dy ** 2
            * (
                self.q_half_plus[1, 1:-1, 1:-1]
                * (self.u1[1:-1, 1:] - self.u1[1:-1, :-1])[:, 1:]
                - self.q_half_minus[1, 1:-1, 1:-1]
                * (self.u1[1:-1, 1:] - self.u1[1:-1, :-1])[:, :-1]
            )
        )

        return ddx + ddy + self.f(self.xv, self.xv, self.dt*self.n)


    def __call__(self):
        """
        Returns last solution without ghost points
        """
        return self.u[1:-1, 1:-1]

    def run_simulation(self):
        self.u_complete = np.zeros((self.Nt, self.Nx, self.Ny))
        for i in trange(self.Nt):
            self.u_complete[i] = self.u[1:-1, 1:-1]
            #self.plot_current_solution(save = False)
            self.advance()

        t1 = time.time()
        return self.u_complete
        # Important to set u = u_1 if u is to be returned!

    def ghost_correction(self, arr):
        arr[0, :] = arr[2, :]
        arr[-1, :] = arr[-3, :]
        arr[:, 0] = arr[:, 2]
        arr[:, -1] = arr[:, -3]
        return arr

    def animator(self, skip, vmin, vmax):
        t = np.linspace(0, self.T, self.Nt)
        skip_idx = np.arange(0, len(t), int(skip))

        # skip frames for speed
        u_complete = self.u_complete[skip_idx]
        t = t[skip_idx]



        fig = plt.figure()
        ax = plt.axes(projection='3d')
        plt.xlabel(r"$x$", fontsize=14)
        plt.ylabel(r"$y$", fontsize=14)
        ax.plot_surface(self.xv, self.yv, u_complete[0], cmap='viridis', vmin=-1, vmax=1)

        def init():
            plt.clf()
            #ax = sns.heatmap(u_complete[0], vmin, vmax)
            ax.plot_surface(self.xv, self.yv, u_complete[0], cmap='viridis', vmin=-1, vmax=1)

        def animate(i):
            plt.clf()
            plt.title(f"t = {t[i]:.1f}/{t[-1]:.1f}")
            #ax = sns.heatmap(u_complete[i], vmin, vmax)
            ax.plot_surface(self.xv, self.yv, u_complete[i], cmap='viridis', vmin=-1, vmax=1)
            #get less numbers on labels
            plt.xlabel(r"$x$", fontsize=14)
            plt.ylabel(r"$y$", fontsize=14)

        ani = animation.FuncAnimation(fig, animate, frames = len(t), init_func=init, interval=1)
        return ani

    def animate(self, show = True, skip = 1, save = False, fps = 30, vmin = np.nan, vmax = np.nan):
        ani = self.animator(skip, vmin, vmax)

        if isinstance(save, str): #save
            save = f"../animations/ani_{save}.gif"
            dpi = 200
            writer = 'imagemagick'
            print(f"#------- saving file: {save} -------#")
            ani.save(save, writer = writer, dpi = 200, fps = fps)
        if show:
            plt.show()


    def plot_current_solution(self, save = False):
        fig = plt.figure()
        plt.pcolormesh(self.xv, self.xv, self.u[1:-1,1:-1], shading = "auto")
        plt.xlabel(r"$x$", fontsize=14)
        plt.ylabel(r"$y$", fontsize=14)
        plt.colorbar()
        plt.tight_layout(pad=1.1, w_pad=0.7, h_pad=0.2)
        if isinstance(save, str): #save
            plt.savefig(f"../animations/fig_{save}.pdf", bbox_inches="tight")

        plt.show()


if __name__ == "__main__":
    def exact_solution(x, y, t):
        return standing_waver(x, y, t)

    def I(x, y):
        return exact_solution(x, y, 0)

    def V(x, y):
        mx = my = m = 1
        kx = m * np.pi / Lx
        ky = m * np.pi / Ly
        w = np.sqrt(kx**2 + ky**2)
        A = 1
        t=0
        return -w*A*np.cos(kx*x)*np.cos(ky*y)*np.sin(w*t)

    def standing_waver(x, y, t):
        mx = my = m = 1
        kx = m * np.pi / Lx
        ky = m * np.pi / Ly
        w = np.sqrt(kx**2 + ky**2)
        A = 1
        return A * np.cos(kx * x) * np.cos(ky * y) * np.cos(w * t)

    def q(x,y):
        return 1

    def f(x,y,t):
        return 0


    b = 0
    Lx, Ly = 1, 1
    dx, dy = 0.5, 0.5
    dt = 0.01
    T = 1

    wave = waveSolver(dx, dy, Lx, Ly, T, dt, I, V, f, q, b)
    wave.run_simulation()
    wave.animate(show = True, skip = 1, save = False)
