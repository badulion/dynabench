import numpy as np
from findiff import coefficients

def differential_operator(derivative: str = "u_x_0_y_0", acc: int = 2, dx: float = 1, dy: float = 1) -> np.ndarray:

    """
        For a given string representation of a derivative, return the corresponding differential operator 
        as a numpy array. The derivative string is formatted as follows: u_x_i_y_j, where i and j are the
        order of the derivative with respect to x and y, respectively.

        Parameters:
        -----------
        derivative : str
            The string representation of the derivative.
        acc : int
            The accuracy of the derivative.
        dx : float
            The spacing between grid points in the x direction.
        dy : float
            The spacing between grid points in the y direction.

        Returns:
        --------
        np.ndarray
            The differential operator as a numpy array.
    """

    # parse orders
    derivative = derivative.split("_")
    order_x, order_y = int(derivative[2]), int(derivative[4])

    if order_x == 0:
        c_x = np.array([[1]])
    else:
        c_x = coefficients(deriv=order_x, acc=acc)['center']['coefficients'].reshape(-1, 1)/dx**order_x

    if order_y == 0:
        c_y = np.array([[1]])
    else:
        c_y = coefficients(deriv=order_y, acc=acc)['center']['coefficients'].reshape(1, -1)/dy**order_y

    return c_x@c_y