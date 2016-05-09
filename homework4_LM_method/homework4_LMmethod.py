
import numpy as np
import matplotlib.pyplot as plt

# example taken from machine learningan algorithmic perspective
# function is f(x_1, x_2) = 100 * (x_2 - x_1^2)^2 + (1 - x_1)^2
# corresponding r(x) = (10(x_2 - x_1^2), 1 - x_1)
def f(p):
    # print(p)
    r = np.array([10 * (p[1] - p[0] ** 2), 1 - p[0]])
    # print(r)
    f_value = r.T.dot(r)
    # print(f_value)
    J = np.array([[-20 * p[0], 10], [-1, 0]])
    # print(J)
    grad = J.T.dot(r.T)
    # print(grad)
    return r, f_value, J, grad


class levMar(object):

    def __init__(self, p0, function, tol=1e-5, max_iter=100):
        self.p0 = p0
        self.tol = tol
        self.max_iter = max_iter
        self.function = function
        self.p_history = []
        self.f_history = []

    # implement its pseudo code here
    def optimize(self):
        n_variables = self.p0.shape[0]
        v = 0.01
        p = self.p0
        r, f_value, J, grad = f(p)
        e = np.sum(r.T.dot(r))
        current_iter = 0
        while current_iter < self.max_iter and np.linalg.norm(grad) > self.tol:
            current_iter += 1
            r, f_value, J, grad = f(p)
            H = J.T.dot(J) + v * np.eye(n_variables)

            p_new = np.zeros(self.p0.shape)
            inner_iter = 0
            while (p != p_new).all() and inner_iter < self.max_iter:
                inner_iter += 1
                dp, resid, rank, s = np.linalg.lstsq(H, grad)
                p_new = p - dp
                r_new, f_value_new, J_new, grad_new = f(p_new)
                e_new = np.sum(r_new.T.dot(r_new))
                actual = np.linalg.norm(r.T.dot(r) - r_new.T.dot(r_new))
                predicted = np.linalg.norm(grad.T.dot(p_new - p))
                rho = actual / predicted
                # print(rho)

                if rho > 0:
                    p = p_new
                    e = e_new
                    if rho > 0.25:
                        v /= 10
                else:
                    v *= 10
            print(f_value, p, e, np.linalg.norm(grad), v)
            self.p_history.append(p)
            self.f_history.append(f_value)

    def plot(self):
        iter = np.arange(len(self.f_history))
        plt.title('Levenberg–Marquardt algorithm result')
        plt.xlabel('iteration')
        plt.ylabel('function value')
        plt.plot(iter, self.f_history, 'bo')
        plt.plot(iter, self.f_history, 'g-')
        plt.show()


if __name__ == '__main__':
    p = np.array([-1, 2])
    # f(p)
    lm_object = levMar(p, f(p))
    lm_object.optimize()
    lm_object.plot()





