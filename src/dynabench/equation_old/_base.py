import re
from pde import PDE
from sympy import Symbol, sympify, Eq, Function, expand

from typing import List, Dict


class BaseEquation(object):
    """
        Base class for all equations. Represents the equation in the form of a dictionary which can be used with the 
        py-pde library.

        Parameters
        ----------
        equations : List[str] | str, default ["u_t = 0"]
            List of equations to be represented.
        parameters : dict, default {}
            Dictionary of parameters for the equations.

        Attributes
        ----------
        linear_terms : List[str]
            List of linear terms in the equation.
        nonlinear_terms : List[str]
            List of nonlinear terms in the equation.

    """
    def __init__(self, 
                 equations: List[str] | str = ["dt(u) = 0"],
                 parameters: Dict[str, float] = {}, 
                 **kwargs):

        self.parameters = parameters   

        # Clean whitespace from the equations
        equations = [re.sub(r"\s+", "", eq, flags=re.UNICODE) for eq in equations]

        # Substitute higher order time derivative equations by reducing them to first order
        self._equations_str = []
        for eq in equations:
            first_order_eqs = self._first_ordify(eq)
            self._equations_str += first_order_eqs

        # extract variables from the equations
        self._variables = list()
        for eq in self._equations_str:
            lhs, rhs = eq.split("=")
            self._variables.append(re.sub(r"dt+\((.+)\)", "\g<1>", lhs, flags=re.UNICODE))

        # Divide the equations into LHS and RHS
        self._rhs_str = list()
        self._lhs_str = list()
        for eq in self._equations_str:
            lhs, rhs = eq.split("=")
            self._lhs_str.append(lhs)
            self._rhs_str.append(rhs)

    
    def _first_ordify(self, equation: str) -> List[str]:
        """
            Convert higher order time derivative equations to first order.

            Parameters
            ----------
            equation : str
                The equation to be converted.

            Returns
            -------
            List[str]
                List of first order time derivative equations.
        """
        # determine order of the eq
        lhs, rhs = equation.split("=")
        order = 0
        lhs_var = re.sub(r"dt+\((.+)\)", "\g<1>", lhs, flags=re.UNICODE)
        order_str = re.sub(r"d(t+)\(.+\)", "\g<1>", lhs, flags=re.UNICODE)
        order = len(order_str)

        if order == 1:
            return [equation]
        else: 
            # reduce order by adding a new variable with reduced order
            new_lhs_var = f"{lhs_var}t"
            new_lhs_time_derivative = f"d{order_str[:-1]}({new_lhs_var})"
            return [f"dt({lhs_var})={new_lhs_var}"] + self._first_ordify(f"{new_lhs_time_derivative}={rhs}")



    def export_as_pypde_equation(self):
        """
            Export the equation as a py-pde equation.
        """
        eq = PDE({var: eq for var, eq in zip(self._variables, self._rhs_str)}, consts=self.parameters)
        return eq


    @property
    def variables(self):
        """
            Get the variables of the equation.
        """
        return self._variables
    
    @property
    def num_variables(self):
        """
            Get the number of variables in the equation.
        """
        return len(self._variables)
    
    @property
    def equations(self):
        """
            Get the equations of the equation.
        """
        return self._equations_str
    
    @property
    def rhs(self):
        """
            Get the right-hand side of the equation.
        """
        return self._rhs_str
    
    @property
    def lhs(self):
        """
            Get the left-hand side of the equation.
        """
        return self._lhs_str


if __name__ == "__main__":

    eq = BaseEquation(equations=["dt(dt(dt(u))) = dxxyy(u)*dx(u**3) + D*dx(u)*(u+dxx(u))"], parameters={"D": 1})