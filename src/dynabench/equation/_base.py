import numpy as np
import re
from sympy import Symbol, sympify, Eq, Function, expand

from typing import List, Dict

class Term(object):
    """
        Represents a term in an equation.

        Parameters
        ----------
        term : str
            The term to be represented.
        linear : bool, default True
            Whether the term is linear. Defaults to True.
    """
    def __init__(self, term: str, linear: bool = True):
        self.term = term
        self.linear = linear

    def __str__(self):
        return self.term

class BaseEquation(object):
    """
        Base class for all equations.

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
    _allowed_operators = ['+', '-', '*', '/', '^', '(', ')']
    _allowed_rhs_diff_operators = r'dx*y*'
    _allowed_lhs = r'dt(.+)'
    def __init__(self, 
                 equations: List[str] | str = ["u_t = 0"],
                 parameters: Dict[str, float] = {}, 
                 **kwargs):

        if type(equations) == str:
            equations = [equations]

        
        # Clean the equations to be parsable
        equations = [re.sub(r"\s+", "", eq, flags=re.UNICODE) for eq in equations]

        # validate the LHS
        lhss = [eq.split("=")[0] for eq in equations]

        for lhs in lhss:
            if not re.fullmatch(BaseEquation._allowed_lhs, lhs):
                raise ValueError("Invalid LHS")
            
        # Substitute higher order time derivative equations by reducing them to first order
        self._equations_str = []
        for eq in equations:
            first_order_eqs = self._first_ordify(eq)
            self._equations_str += first_order_eqs

        # Divide the equations into LHS and RHS
        self._lhs_str, self._rhs_str = zip(*[eq.split("=") for eq in self._equations_str])

        # expand rhs of the equation to monomial terms
        self._rhs_str = [str(expand(rhs)) for rhs in self._rhs_str]
        self._equations_str = [f"{lhs}={rhs}" for lhs, rhs in zip(self._lhs_str, self._rhs_str)]
            


        # Collect the differential operators in the equation
        self.diff_operators = []
        for rhs in self._rhs_str:
            self.diff_operators += self._find_diff_operators(rhs)

        # validate the variables from lhs and rhs
        variables_lhs = [re.sub(r"dt\((.*)\)", "\g<1>", eq, flags=re.UNICODE) for eq in self._lhs_str]
        variables_rhs = []
        for eq in self._rhs_str:
            variables_rhs += self._extract_variables_rhs(eq)
        variables_rhs = set(variables_rhs)
        variables_rhs = variables_rhs - set(parameters.keys())

        if not set(variables_lhs) == set(variables_rhs):
            raise ValueError("The variables in the LHS and RHS must be the same.")

        self._variables = list(set(variables_lhs))

        if len(self._equations_str) != len(self._variables):
            raise ValueError("The number of equations and variables must be the same.")

        for eq in self._equations_str:
            self._validate_equation(eq)


    def __str__(self):
        return self._equations_str
    
    def _extract_variables_rhs(self, rhs: str) -> List[str]:
        """
            Extract the variables from the right-hand side of the equation.

            Parameters
            ----------
            rhs : str
                The right-hand side of the equation.

            Returns
            -------
            List[str]
                List of variables in the equation.
        """
        return [str(symbol) for symbol in sympify(rhs).free_symbols]
    
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
        lhs_var = lhs
        while re.match(r"dt\((.*)\)", lhs_var):
            order += 1
            lhs_var = re.sub(r"dt\((.*)\)", "\g<1>", lhs_var, flags=re.UNICODE)

        if order == 1:
            return [equation]
        else: 
            # reduce order by adding a new variable with reduced order
            new_lhs_var = f"{lhs_var}t"
            new_lhs_time_derivative = "dt("*(order-1)+new_lhs_var+")"*(order-1)
            return [f"dt({lhs_var})={new_lhs_var}"] + self._first_ordify(f"{new_lhs_time_derivative}={rhs}")


    def _find_diff_operators(self, rhs_expr: str) -> List[str]:
        """
            Find the differential operators in the equation.

            Parameters
            ----------
            rhs_expr : str
                The right-hand side of the equation.

            Returns
            -------
            List[str]
                List of differential operators in the equation.
        """
        diff_operators = []
        for operator in [self._allowed_rhs_diff_operators]:
            diff_operators += re.findall(operator, rhs_expr)
        return diff_operators
    
    def _validate_equation(self, equation_str: str):
        """
            Validate the equation string.

            Parameters
            ----------
            equation : str
                The string representation of the equation.

            Returns
            -------
            None

            Raises
            ------
            ValueError
                If the equation string is invalid.
        """

        try:
            lhs, rhs = equation_str.split("=")
            Eq(sympify(lhs), sympify(rhs))
        except:
            raise ValueError("Invalid equation string.")

    @property
    def variables(self):
        """
            Get the variables of the equation.
        """
        return self._variables

if __name__ == "__main__":

    eq = BaseEquation(equations=["dt(dt(dt(u))) = dxxyy(u)*dx(u**3) + D*dx(u)*(u+dxx(u))"], parameters={"D": 1})