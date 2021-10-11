import numpy as np
import scipy.integrate as scp_i
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class DoublePendulum:
    """
    Klassen beskriver en dobbel pendel
    """
    def __init__(self, M1 = 1, L1 = 1, M2 = 1, L2 = 1, g = 9.81):
        self.M1 = M1
        self.L1 = L1
        self.M2 = M2
        self.L2 = L2
        self.g = g
        self._t = None
        self._theta1 = None
        self._theta2 = None
        self._omega1 = None
        self._omega2 = None
        self._x1 = None
        self._u1 = None
        self._x2 = None
        self._u2 = None
        self._potential = None
        self._vx1 = None
        self._vy1 = None
        self._vx2 = None
        self._vy2 = None
        self._kinetic = None

    def __call__(self, t, y):
        """
        - solve_ivp tar inn y'(t) som første argumentog  kaller derfor
          på objektet (self) da __call__ metoden returnerer y'(t).
        """
        g = self.g
        theta1 = y[0]; theta2 = y[2]
        omg1 = y[1]; omg2 = y[3]
        delta_theta = theta2 - theta1
        M1, L1, M2, L2 = self.M1, self.L1, self.M2, self.L2
        o1_ledd1 = M2*L1*(omg1**2)*np.sin(delta_theta)*np.cos(delta_theta)
        o1_ledd2 = M2*g*np.sin(theta2)*np.cos(delta_theta)
        o1_ledd3 = M2*L2*(omg2**2)*np.sin(delta_theta)
        o1_ledd4 = -(M1 + M2)*g*np.sin(theta1)
        o1_ledd5 = (M1 + M2)*L1
        o1_ledd6 = -M2*L1*(np.cos(delta_theta))**2
        dw1_dt = (o1_ledd1 + o1_ledd2 + o1_ledd3 + o1_ledd4)/(o1_ledd5 + o1_ledd6)
        o2_ledd1 = -M2*L2*(omg2**2)*np.sin(delta_theta)*np.cos(delta_theta)
        o2_ledd2 = (M1 + M2)*g*np.sin(theta1)*np.cos(delta_theta)
        o2_ledd3 = -(M1 + M2)*L1*(omg1**2)*np.sin(delta_theta)
        o2_ledd4 = -(M1 + M2)*g*np.sin(theta2)
        o2_ledd5 = (M1 + M2)*L2
        o2_ledd6 = -M2*L2*(np.cos(delta_theta))**2
        dw2_dt = (o2_ledd1 + o2_ledd2 + o2_ledd3 + o2_ledd4)/(o2_ledd5 + o2_ledd6)
        dy_dt = tuple([omg1, dw1_dt, omg2, dw2_dt])
        return dy_dt

    def solve(self, y0, T, dt, angles = "rad"):
        """
        - Metoden tar blant annet inn tuppelen y0 = (theta1,omega1,theta2,omega2)
          som initialverdier av hhv. vinkelen og hastigheten til pendel 1 og 2.
        - Om theta oppgis i grader må angles settes til "deg" for at metoden skal fungere.
        - Metoden lagrer til slutt t, theta 1 og 2, omega 1 og 2 og dt internt i klassen.
        """
        if angles == "deg":
            y0[0], y0[2] = y0[0]*np.pi/180, y0[2]*np.pi/180
        t_span = np.linspace(0, T, (int(T/dt) + 1))
        solver = scp_i.solve_ivp(self, [0,T], y0, t_eval = t_span)
        self._t, self.y = solver.t, solver.y
        self._theta1, self._theta2 = self.y[0], self.y[2]
        self._omega1, self._omega2 = self.y[1], self.y[3]
        self.dt = dt

    def solveexce(self, var):
        if var is None:
            raise Exception("Solve has not been called")
        else:
            return var

    @property
    def t(self):
        return self.solveexce(self._t)
    @property
    def theta1(self):
        return self.solveexce(self._theta1)
    @property
    def theta2(self):
        return self.solveexce(self._theta2)
    @property
    def x1(self):
        return self.L1*np.sin(self._theta1)
    @property
    def u1(self):
        return -self.L1*np.cos(self._theta1)
    @property
    def x2(self):
        return self.x1 + self.L2*np.sin(self._theta2)
    @property
    def u2(self):
        return self.u1 - self.L2*np.cos(self._theta2)
    @property
    def omega1(self):
        return self.solveexce(self._omega1)
    @property
    def omega2(self):
        return self.solveexce(self._omega2)
    @property
    def potential(self):
        P = self.M1*self.g*(self.u1 + self.L1)
        P += self.M2*self.g*(self.u2 + self.L1 + self.L2)
        return P
    @property
    def vx1(self):
            return np.gradient(self.x1, self.t)
    @property
    def vy1(self):
            return np.gradient(self.u1, self.t)
    @property
    def vx2(self):
            return np.gradient(self.x2, self.t)
    @property
    def vy2(self):
            return np.gradient(self.u2, self.t)
    @property
    def kinetic(self):
        K = (1/2)*self.M1*((self.vx1)**2 + (self.vy1)**2)
        K += (1/2)*self.M2*((self.vx2)**2 + (self.vy2)**2)
        return K

    def create_animation(self):
        """
        - Definerer en metode for å håndtere animasjonen av den doble pendelen.
        """
        fig = plt.figure()

        plt.axis('equal')
        plt.axis('off')
        plt.axis((-3, 3, -3, 3))

        self.pendulums, = plt.plot([], [], 'o-')

        self.animation = FuncAnimation(fig,
                                                 self._next_frame,
                                                 frames=range(len(self.x1)),
                                                 repeat=None,
                                                 interval=1000*self.dt,
                                                 blit=True)

    def _next_frame(self, i):
        self.pendulums.set_data((0, self.x1[i], self.x2[i]),
                                (0, self.u1[i], self.u2[i]))
        return self.pendulums,

    def show_animation(self):
        plt.show()

    def save_animation(self):
        self.animation.save("pendulum_motion.mp4", fps=60)

def example_run():
    """
    - Definerer en funksjon for å lage en instans av DoublePendulum,
      løse ODE settet med passende initialverdier og plotte den kinetiske, potensielle
      og totale energien til den doble pendelen.
    - Skal også lage og vise animasjonen.
    """
    obj = DoublePendulum()
    obj.solve((np.pi/2,0,np.pi/2,0), 10, 0.1)

    plt.plot(obj.t, obj.kinetic)
    plt.plot(obj.t, obj.potential)
    plt.plot(obj.t, obj.potential + obj.kinetic)
    plt.show()

    obj.create_animation()
    obj.show_animation()

if __name__ == "__main__":
    example_run()
