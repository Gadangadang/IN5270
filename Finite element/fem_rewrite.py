import numpy as np
import matplotlib.pyplot as plt


class FEM:
    def __init__(self, C, D, f, exact, elements):
        self.C = C
        self.D = D
        self.elements = elements
        self.place = 1 / elements

        # Basis functions for P2 and P1
        self.phi_0_quad = lambda X: 0.5 * (X - 1) * X
        self.phi_1_quad = lambda X: 1 - X ** 2
        self.phi_2_quad = lambda X: 0.5 * (X + 1) * X
        self.phi_0_lin = lambda X: 0.5 * (1 - X)
        self.phi_1_lin = lambda X: 0.5 * (1 + X)

        self.quad = [self.phi_0_quad, self.phi_1_quad, self.phi_2_quad]
        self.lin = [self.phi_0_lin, self.phi_1_lin]

        # External functions
        self.f = f
        self.exact = exact

    def x_val(self, e, X):
        return self.xm[e] + h / 2 * X

    def integrate(self, f1, f2, N, e):
        X = np.linspace(-1, 1, N + 1)
        if f1 == "f":
            f1x = f1(self.x_val(e, X))
        elif f2 == "f":
            f2x = f2(self.x_val(e, X))
        else:
            f1x = f1(X)
            f2x = f2(X)
        area = np.sum(f1x * f2x) * self.h / N
        return area

    def create_A_matrix(self):
        """
        **************************************************
        This function creates the A matrix, used in
        the matrix equation Au = b. The values for each
        A_ij has been precalculated, but can be calculated
        using symbolic integration for more freedom. The
        matrix is then inverted, so one can later find
        the vector u.
        **************************************************
        """
        # Calculate element placement along domain
        self.element_placement = []
        start = 0
        for i in range(1, self.elements + 1):
            self.element_placement.append([start, i * self.place])
            start = i * self.place

        # print(self.element_placement)

        self.xm = np.zeros(len(self.element_placement))
        x_arr = [0]

        for i, elem in enumerate(self.element_placement):
            self.xm[i] = 0.5 * (elem[0] + elem[1])
            if i < int(self.elements / 2):
                x_arr.append(self.xm[i])
                x_arr.append(elem[1])
            else:
                x_arr.append(elem[1])

        self.x_arr = np.array(x_arr)

        self.xm = np.asarray(self.xm)
        self.h = np.max(self.x_arr) / self.elements

        self.nodes = int(3 / 2 * self.elements + 1)
        A = np.zeros((self.nodes - 1, self.nodes - 1))

        # Precalculated matrix elements
        Atilde_0 = np.array([[7, -8, 1], [-8, 16, -8], [1, -8, 7]])
        Atilde_1 = np.array([[3, -3], [-3, 3]])

        # Matrix assembly
        for e in range(0, self.elements, 2):
            for i in range(3):
                for j in range(3):
                    A[e + i, e + j] += Atilde_0[i, j]

        for e in range(int(self.elements), self.nodes - 2):
            for i in range(2):
                for j in range(2):
                    A[e + i, e + j] += Atilde_1[i, j]


        A[-1, -1] += 3 #Add last contribution

        self.A = (1 / (3 * self.h)) * A

        self.A_inv = np.linalg.inv(self.A)

    def create_B_vector(self):
        """
        *********************************************
        Calculate the b vector in the matrix equation
        Au = b. First create a degree of freedom map
        to keep track of which node that contributes
        for each element. Lastly there is vector
        assembly, and corrections from boundary terms
        *********************************************
        """

        b = np.zeros(self.nodes - 1)  # holds the b vector elements, will be assembled

        mid = int(self.elements / 2)
        dof_map = []
        last_i = 2
        for e in range(1, self.elements + 1):
            dof = []
            if e == 1:
                for i in range(3):
                    dof.append(i)
            elif (e <= mid) and (e != 1):
                for i in range(3):
                    dof.append(last_i + i)
            elif (e >= mid) and (e != self.elements):
                for i in range(2):
                    dof.append(last_i + i)
            else:
                dof.append(last_i)
            last_i = dof[-1]
            dof_map.append(dof)

        # print(dof_map)

        for elem in range(int(self.elements / 2)):
            b_index = dof_map[elem]
            for i, ind in enumerate(b_index):
                b[ind] += self.integrate(f, self.quad[i], 10001, elem)

        for elem in range(int(self.elements / 2), self.elements):
            b_index = dof_map[elem]
            for i, ind in enumerate(b_index):
                b[ind] += self.integrate(f, self.lin[i], 10001, elem)

        self.dof_map = dof_map
        self.b = b
        self.b[0] -= self.C

        self.b[-1] -= (
            self.D * 2 / self.h * (-1 / 2)
        )  # Integrate D phi´_N * phi´_i = 1/jacobi_det *(-1/2)
        # print(b)

    def final_solution(self):
        """
        **********************************************
        Computes the constants for the basis functions
        and constructs the numerical approximation.
        It then plots the solution against the exact
        solution.
        **********************************************
        """

        constants = self.A_inv @ self.b
        constants = np.append(constants, self.D)
        # print(constants)

        # Show final solution
        solution = []
        X = np.linspace(-1, 1, 101)
        for elem in range(int(self.elements / 2)):
            b_index = self.dof_map[elem]
            sol = np.zeros(len(X))
            for i, ind in enumerate(b_index):
                sol += constants[ind] * self.quad[i](X)
            solution.append(sol)

        for elem in range(int(self.elements / 2), self.elements):
            b_index = self.dof_map[elem]
            sol = np.zeros(len(X))
            for i, ind in enumerate(b_index):
                sol += constants[ind] * self.lin[i](X)
            if elem == (self.elements - 1):
                sol += constants[-1] * self.lin[-1](X)
            solution.append(sol)

        self.constants = constants
        solution = np.concatenate(solution)
        self.solution = solution

        x_data = np.linspace(0, 1, len(solution))

        plt.plot(x_data, self.exact(x_data, self.C, self.D), label="Exact")
        plt.plot(
            x_data,
            self.solution,
            label="Approx C: {:.1f} D: {:.1f}".format(self.C, self.D),
        )
        plt.legend()
        plt.show()


    def eval_func(self, coord):

        #Find the element where the coordinate lies in
        elements = np.mean(self.element_placement, axis=1)
        #print(elements)
        elem = np.argmin(np.abs(elements - coord))
        b_index = self.dof_map[elem]

        sol = 0
        X_coor = 2/self.h*(coord - self.xm[elem])

        if elem > self.elements:
            self.p = self.lin #Use linear basis functions
        else:
            self.p = self.quad #Use quadratic basis functions
        for i, ind in enumerate(b_index):
            sol += self.constants[ind] * self.p[i](X_coor)
        if elem == (self.elements - 1):
            sol += self.constants[-1] * self.lin[-1](X_coor)

        return sol

def f(x):
    return 2


def exact(x, C, D):
    return C * x + D + 1 - C - x ** 2


if __name__ == "__main__":
    print("Test OO code for FEM")
    D = 1.4
    C = 0.3

    FEM_test = FEM(C, D, f, exact, 4)
    FEM_test.create_A_matrix()
    FEM_test.create_B_vector()
    FEM_test.final_solution()

    print(FEM_test.eval_func(0.5))
