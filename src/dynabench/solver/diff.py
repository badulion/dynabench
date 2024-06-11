import numpy as np
import findiff
import scipy.signal
import re

# ToDo: implement boundary conditions
class DifferentialOperator(object):
    """
        Represents a differential operator by a finite difference method. The internal representation 
        of the operator is a numpy array and the operator is applied by running a convolution.
        See `NeuralPDE: Modelling Dynamical Systems from Data <https://arxiv.org/abs/2111.07671>`_ for more details.

        The coefficients of the operator are computed using the `findiff <https://pypi.org/project/findiff/>`_ package.

        Parameters
        ----------
        derivative : str, default "dx"
            The string representation of the derivative.
        acc : int, default 2
            The accuracy with which to calculate the derivative.
        dx : float, default 1
            The spacing between grid points in the x direction.
        dy : float, default 1
            The spacing between grid points in the y direction.

        Methods
        -------
        __call__(u: np.ndarray) -> np.ndarray
            Apply the differential operator to the input array.
    """

    def __init__(self, derivative: str = "dx", acc: int = 2, dx: float = 1, dy: float = 1):
        self.derivative = derivative
        self.acc = acc
        self.dx = dx
        self.dy = dy

        self.coeffs = self._calculate_coeffs(derivative, acc, dx, dy)

    def _calculate_coeffs(self, derivative_str: str, acc: int, dx: float, dy: float) -> np.ndarray:
        """
            Calculate the coefficients of the differential operator.

            Parameters
            ----------
            derivative_str : str
                The string representation of the derivative.
            acc : int
                The accuracy with which to calculate the derivative.
            dx : float
                The spacing between grid points in the x direction.
            dy : float
                The spacing between grid points in the y direction.

            Returns
            -------
            np.ndarray
                The coefficients of the differential operator.
        """
        # parse orders
        if not re.fullmatch(r"dx*y*", derivative_str):
            raise ValueError("The derivative string should be of the form 'dx*y*' where x and y are repeated for the order of the derivative, e.g. 'dxyy'.")

        order_x, order_y = derivative_str.count('x'), derivative_str.count('y')

        if order_x == 0:
            c_x = np.array([[1]])
        else:
            c_x = findiff.coefficients(deriv=order_x, acc=acc)['center']['coefficients'].reshape(-1, 1)/dx**order_x

        if order_y == 0:
            c_y = np.array([[1]])
        else:
            c_y = findiff.coefficients(deriv=order_y, acc=acc)['center']['coefficients'].reshape(1, -1)/dy**order_y

        return c_x@c_y

    def __str__(self):
        return f"{self.derivative}"

    def __call__(self, u: np.ndarray) -> np.ndarray:
        return scipy.signal.convolve2d(u, self.coeffs, mode="same", boundary="wrap")
