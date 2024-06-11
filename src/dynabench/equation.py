"""
Module for representing partial differential equations.
"""

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
                 evolution_rate: float = 1.0,
                 **kwargs):

        self.parameters = parameters   
        self.evolution_rate = evolution_rate

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
            self._variables.append(self._extract_variable_from_lhs(lhs))

        # Divide the equations into LHS and RHS
        self._rhs_str = list()
        self._lhs_str = list()
        for eq in self._equations_str:
            lhs, rhs = eq.split("=")
            self._lhs_str.append(lhs)
            self._rhs_str.append(rhs)

    @property
    def name(self):
        """
            Get the name of the equation.
        """
        return self.__class__.__name__.lower().removesuffix("equation")
    
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
        lhs_var = self._extract_variable_from_lhs(lhs)
        order_str = re.sub(r"d(t+)\(.+\)", r"\g<1>", lhs, flags=re.UNICODE)
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
        eq = PDE(rhs={var: f"{self.evolution_rate}*({eq})" for var, eq in zip(self._variables, self._rhs_str)}, 
                 consts=self.parameters)
        return eq
    
    def simplify_equation(self, eq: str) -> str:
        """
            Simplify the equation.

            Parameters
            ----------
            eq : str
                The equation to be simplified.

            Returns
            -------
            str
                The simplified equation.
        """

        def _parse_derivative_expr(expr: str):
            """
                Parse the derivative expression.

                Parameters
                ----------
                expr : str
                    The derivative expression of the form represented in a sympy expr.
                    Examples include:
                    - Derivative(u(x, y), (x, 4))
                    - Derivative(u(x, y), x, (y, 2))
                    - Derivative(u(x, y), x, y), 
                    - Derivative(u(x, y), (y, 2))
                    - Derivative(u(x, y), (x, 2), (y, 2))

                Returns
                -------
                str
                    The parsed derivative expression. So the above expressions will be parsed:
                    - Derivative(u(x, y), (x, 4)) -> dxxxx(u)
                    - Derivative(u(x, y), x, (y, 2)) -> dxyy(u)
                    - Derivative(u(x, y), x, y) -> dxy(u)
                    - Derivative(u(x, y), (y, 2)) -> dyy(u)
                    - Derivative(u(x, y), (x, 2), (y, 2)) -> dxxdyy(u)
            """
            
            expr = re.sub(r"\s+", "", expr, flags=re.UNICODE) # remove all whitespaces
            var = re.search(r"Derivative\((\w+)\(", expr).group(1)
            variable_counts = sympify(expr).variable_count
            return "d"+"".join([f"{k[0]}"*k[1] for k in variable_counts])+f"({var})"
        lhs, rhs = eq.split("=")
        var = self._extract_variable_from_lhs(lhs)

        x, y = Symbol('x'), Symbol('y')
        f = Function('u')(x, y)
        def laplace(u):
            return u
        
        print(rhs)
        
        return sympify(rhs, locals={'laplace': laplace, 'u': f})
                   
    def _extract_variable_from_lhs(self, lhs: str) -> str:
        """
            Extract the variable from the left-hand side of the equation.

            Parameters
            ----------
            lhs : str
                The left-hand side of the equation.

            Returns
            -------
            str
                The variable in the equation.
        """
        return re.sub(r"dt+\((.+)\)", r"\g<1>", lhs, flags=re.UNICODE)


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
    
class AdvectionEquation(BaseEquation):
    """
        Advection equation in 2D. The equation is given by:

        .. math::
            \\frac{\\partial u}{\\partial t} = -c_x \\frac{\\partial u}{\\partial x} - c_y \\frac{\\partial u}{\\partial y}

        where c_x and c_y are the speeds of the advection in the x and y directions respectively.

        Parameters
        ----------
        parameters : dict, default {c_x: 1, c_y: 1}
            Dictionary of parameters for the equations.

        Attributes
        ----------
        linear_terms : List[str]
            List of linear terms in the equation.
        nonlinear_terms : List[str]
            List of nonlinear terms in the equation.

    """
    def __init__(self, c_x: float = 1.0, c_y: float = 1.0, evolution_rate: float = 1.0, **kwargs):
        parameters = {'c_x': c_x, 'c_y': c_y}
        super().__init__(equations=["dt(u) = -c_x*d_dx(u)-c_y*d_dy(u)"], parameters=parameters, evolution_rate=evolution_rate, **kwargs)

class WaveEquation(BaseEquation):
    """
        Wave equation in 2D. The equation is given by:

        .. math::
            \\frac{\\partial^2 u}{\\partial t^2} = c^2 \\nabla^2 u

        where c is the speed of the wave.

        Parameters
        ----------
        parameters : dict, default {c: 1}
            Dictionary of parameters for the equations.

        Attributes
        ----------
        linear_terms : List[str]
            List of linear terms in the equation.
        nonlinear_terms : List[str]
            List of nonlinear terms in the equation.

    """
    def __init__(self, c: float = 1.0, evolution_rate: float = 1.0, **kwargs):
        parameters = {'c': c}
        super().__init__(equations=["dtt(u) = c**2*laplace(u)"], parameters=parameters, evolution_rate=evolution_rate, **kwargs)


class CahnHilliardEquation(BaseEquation):
    """
        Cahn-Hilliard equation in 2D. The equation is given by:

        .. math::
            \\frac{\\partial u}{\\partial t} = D \\nabla^2(u^3 - u - \\gamma \\nabla^2(u))

        where D and gamma are parameters of the equation.

        Parameters
        ----------
        parameters : dict, default {D: 1, gamma: 1}
            Dictionary of parameters for the equations.

        Attributes
        ----------
        linear_terms : List[str]
            List of linear terms in the equation.
        nonlinear_terms : List[str]
            List of nonlinear terms in the equation.

    """
    def __init__(self, D: float = 1.0, gamma: float = 1.0, evolution_rate: float = 1.0, **kwargs):
        parameters = {'D': D, 'gamma': gamma}
        super().__init__(equations=["dt(u) = D*laplace(u**3 - u - gamma*laplace(u))"], parameters=parameters, evolution_rate=evolution_rate, **kwargs) 
    

class DiffusionEquation(BaseEquation):
    """
        Diffusion equation in 2D. The equation is given by:

        .. math::
            \\frac{\\partial u}{\\partial t} = D \\nabla^2 u

        where D is the diffusion coefficient.

        Parameters
        ----------
        parameters : dict, default {D: 1}
            Dictionary of parameters for the equations.

        Attributes
        ----------
        linear_terms : List[str]
            List of linear terms in the equation.
        nonlinear_terms : List[str]
            List of nonlinear terms in the equation.

    """
    def __init__(self, D: float = 1.0, evolution_rate: float = 1.0, **kwargs):
        parameters = {'D': D}
        super().__init__(equations=["dt(u) = D*laplace(u)"], parameters=parameters, evolution_rate=evolution_rate, **kwargs)


class FitzhughNagumoEquation(BaseEquation):
    """
        FitzHugh–Nagumo model with diffusive coupling given by the following equations:

        .. math::
            \\frac{\\partial v}{\\partial t} = \\nabla^2 v + v - \\frac{v^3}{3} - w + stimulus
            \\frac{\\partial w}{\\partial t} = \\frac{v + a - b * w}{\\tau}

        where v is the membrane potential and w is the recovery variable, and stimulus, a, b, and tau are parameters of the equation.
        
        
        
    """

    def __init__(self, stimulus: float = 0.5, τ: float = 10, a: float = 0, b: float = 0, evolution_rate: float = 1.0, **kwargs):
        parameters = {'stimulus': stimulus, 'τ': τ, 'a': a, 'b': b}
        super().__init__(equations=["dt(v) = laplace(v) + v - v**3 / 3 - w + stimulus", 
                                    "dt(w) = (v + a - b * w) / τ"], 
                         parameters=parameters,
                         evolution_rate=evolution_rate,
                         **kwargs)