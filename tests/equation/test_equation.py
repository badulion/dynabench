import pytest
from dynabench.equation import (
    BaseEquation,
    AdvectionEquation,
    WaveEquation,
    CahnHilliardEquation,
    DiffusionEquation,
    FitzhughNagumoEquation,
    SimpleBurgersEquation,
    KuramotoSivashinskyEquation
)
from pde import PDE

def test_base_equation_init(custom_base_equation):
    assert custom_base_equation.parameters == {"param1": 1.0}
    assert custom_base_equation.evolution_rate == 1.0
    assert custom_base_equation.equations == ["dt(u)=-u"]
    assert custom_base_equation.variables == ["u"]
    assert custom_base_equation.rhs == ["-u"]
    assert custom_base_equation.lhs == ["dt(u)"]

def test_base_equation_default_init(default_base_equation):
    assert default_base_equation.parameters == {}
    assert default_base_equation.evolution_rate == 1.0
    assert default_base_equation.equations == ["dt(u)=0"]
    assert default_base_equation.variables == ["u"]
    assert default_base_equation.rhs == ["0"]
    assert default_base_equation.lhs == ["dt(u)"]

def test_base_equation_name(default_base_equation):
    assert default_base_equation.name == "base"

def test_base_equation_first_ordify(default_base_equation):
    result = default_base_equation._first_ordify("dtt(u)=c**2*laplace(u)")
    assert result == ["dt(u)=ut", "dt(ut)=c**2*laplace(u)"]

def test_base_equation_export_as_pypde_equation(custom_base_equation):
    pypde_eq = custom_base_equation.export_as_pypde_equation()
    assert isinstance(pypde_eq, PDE)

## TODO: Fix this test
def test_base_equation_simplify_equation(default_base_equation):
    simplified_eq = default_base_equation.simplify_equation("dt(u) = laplace(u)")
    assert simplified_eq is not None

def test_base_equation_extract_variable_from_lhs(default_base_equation):
    variable = default_base_equation._extract_variable_from_lhs("dt(u)")
    assert variable == "u"

def test_base_equation_variables(base_equation_two_variables):
    assert base_equation_two_variables.variables == ["u", "v"]

def test_base_equation_num_variables(base_equation_two_variables):
    assert base_equation_two_variables.num_variables == 2

def test_base_equation_rhs(base_equation_two_variables):
    assert base_equation_two_variables.rhs == ["-u", "-v"]

def test_base_equation_lhs(base_equation_two_variables):
    assert base_equation_two_variables.lhs == ["dt(u)", "dt(v)"]

def test_invalid_equation():
    with pytest.raises(ValueError):
        BaseEquation(equations=["dt(u) = dt(u =)"])

def test_advection_equation_init():
    eq = AdvectionEquation(c_x=2.0, c_y=3.0)
    assert eq.parameters == {"c_x": 2.0, "c_y": 3.0}
    assert eq.equations == ["dt(u)=-c_x*d_dx(u)-c_y*d_dy(u)"]

def test_wave_equation_init():
    eq = WaveEquation(c=2.0)
    assert eq.parameters == {"c": 2.0}
    assert eq.equations == ["dt(u)=ut", "dt(ut)=c**2*laplace(u)"]

def test_cahn_hilliard_equation_init():
    eq = CahnHilliardEquation(D=2.0, gamma=3.0)
    assert eq.parameters == {"D": 2.0, "gamma": 3.0}
    assert eq.equations == ["dt(u)=D*laplace(u**3-u-gamma*laplace(u))"]

def test_diffusion_equation_init():
    eq = DiffusionEquation(D=2.0)
    assert eq.parameters == {"D": 2.0}
    assert eq.equations == ["dt(u)=D*laplace(u)"]

def test_fitzhugh_nagumo_equation_init():
    eq = FitzhughNagumoEquation(stimulus=0.5, τ=10, a=0, b=0)
    assert eq.parameters == {"stimulus": 0.5, "τ": 10, "a": 0, "b": 0}
    assert eq.equations == ["dt(v)=laplace(v)+v-v**3/3-w+stimulus", "dt(w)=(v+a-b*w)/τ"]

def test_simple_burgers_equation_init():
    eq = SimpleBurgersEquation(nu=2.0)
    assert eq.parameters == {"nu": 2.0}
    assert eq.equations == ["dt(u)=-u*(d_dx(u)+d_dy(u))+nu*laplace(u)"]

def test_kuramoto_sivashinsky_equation_init():
    eq = KuramotoSivashinskyEquation()
    assert eq.parameters == {}
    assert eq.equations == ["dt(u)=-laplace(u)-laplace(laplace(u))-gradient_squared(u)"]