from dynabench.generator.initial import sum_of_gaussians,random_uniform,random_normal
import dedalus.public as d3
import numpy as np

def build_problem_advection(grid, speed_x: float = 2, speed_y: float = 2, rate:float=1,  random_state=None):

    # Fields
    u = grid.dist.Field(name='u', bases=(grid.xbasis, grid.ybasis))

    # initial condition
    u['g'] = sum_of_gaussians(grid_size=grid.grid_size, components=5, random_state=random_state)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, grid.coords['x'])
    dy = lambda A: d3.Differentiate(A, grid.coords['y'])

    # Problem
    problem = d3.IVP([u], namespace=locals())

    # Add main equation, with linear terms on the LHS and nonlinear terms on the RHS
    LHS = f"{1/rate}*dt(u)+{speed_x}*dx(u)+{speed_y}*dy(u)"
    RHS = "0"
    problem.add_equation(LHS+"="+RHS)

    return problem

def build_problem_burgers(grid, viscosity: float = 0.1, rate:float=1, random_state=None):

    # Fields
    u = grid.dist.Field(name='u', bases=(grid.xbasis, grid.ybasis))
    v = grid.dist.Field(name='v', bases=(grid.xbasis, grid.ybasis))

    # initial condition
    u['g'] = sum_of_gaussians(grid_size=grid.grid_size, components=5, random_state=random_state)
    v['g'] = sum_of_gaussians(grid_size=grid.grid_size, components=5, random_state=random_state)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, grid.coords['x'])
    dy = lambda A: d3.Differentiate(A, grid.coords['y'])

    # Problem
    problem = d3.IVP([u, v], namespace=locals())

    # Add main equation, with linear terms on the LHS and nonlinear terms on the RHS
    # first part:
    LHS = f"{1/rate}*dt(u) - {viscosity}*lap(u)"
    RHS = f"-u*dx(u)-v*dy(u)"
    problem.add_equation(LHS+"="+RHS)

    # second part:
    LHS = f"{1/rate}*dt(v) - {viscosity}*lap(v)"
    RHS = f"-u*dx(v)-v*dy(v)"
    problem.add_equation(LHS+"="+RHS)

    return problem

def build_problem_gasdynamics(grid, 
                              gamma: float=2, 
                              k: float=0.1, 
                              M: float=2, 
                              mu:float = 0.01, 
                              rate:float=1, 
                              random_state=None):

    # Fields
    p = grid.dist.Field(name='density', bases=(grid.xbasis, grid.ybasis))
    T = grid.dist.Field(name='temperature', bases=(grid.xbasis, grid.ybasis))
    vx = grid.dist.Field(name='velocity_x', bases=(grid.xbasis, grid.ybasis))
    vy = grid.dist.Field(name='velocity_y', bases=(grid.xbasis, grid.ybasis))

    # initial condition
    p['g'] = sum_of_gaussians(grid_size=grid.grid_size, components=5, zero_level=2, random_state=random_state)
    T['g'] = sum_of_gaussians(grid_size=grid.grid_size, components=5, zero_level=2, random_state=random_state)
    vx['g'] = np.zeros(grid.grid_size)
    vy['g'] = np.zeros(grid.grid_size)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, grid.coords['x'])
    dy = lambda A: d3.Differentiate(A, grid.coords['y'])

    # Problem
    problem = d3.IVP([p, T, vx, vy], namespace=locals())

    # Add main equation, with linear terms on the LHS and nonlinear terms on the RHS
    # first part:
    LHS = f"{1/rate}*dt(p)"
    RHS = f"- vx*dx(p) - vy*dy(p) - p*(dx(vx) + dy(vy))"
    problem.add_equation(LHS+"="+RHS)

    # second part:
    LHS = f"{1/rate}*dt(T)"
    RHS = f"- vx*dx(T) - vy*dy(T) - {gamma}*T*(dx(vx) + dy(vy)) + {gamma}*{k}/p * lap(T)"
    problem.add_equation(LHS+"="+RHS)

    # third part:
    LHS = f"{1/rate}*dt(vx)"
    RHS = f"- vx*dx(vx) - vy*dy(vx) - dx(p*T/{M}) + {mu}/p * (dx(dx(vx)) + dx(dy(vy)))"
    problem.add_equation(LHS+"="+RHS)

    # fourth part:
    LHS = f"{1/rate}*dt(vy)"
    RHS = f"- vx*dx(vx) - vy*dy(vx) - dy(p*T/{M}) + {mu}/p * (dy(dy(vy)) + dx(dy(vx)))"
    problem.add_equation(LHS+"="+RHS)

    return problem

def build_problem_kuramotosivashinsky(grid, 
                                      rate:float=1, 
                                      random_state=None):

    # Fields
    u = grid.dist.Field(name='u', bases=(grid.xbasis, grid.ybasis))

    # initial condition
    u['g'] = random_uniform(grid_size=grid.grid_size, random_state=random_state)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, grid.coords['x'])
    dy = lambda A: d3.Differentiate(A, grid.coords['y'])

    # Problem
    problem = d3.IVP([u], namespace=locals())

    # Add main equation, with linear terms on the LHS and nonlinear terms on the RHS
    LHS = f"{1/rate}*dt(u)+lap(u)+lap(lap(u))"
    RHS = "-0.5*(dx(u)**2+dy(u)**2)"
    problem.add_equation(LHS+"="+RHS)

    return problem

def build_problem_reactiondiffusion(grid, 
                                    diffusivity_1: float = 0.1, 
                                    diffusivity_2: float = 0.001, 
                                    k: float = 0.005,
                                    reaction_speed_1: float = 1,
                                    reaction_speed_2: float = 1,
                                    rate:float=1, 
                                    random_state=None):
    # Fields
    u = grid.dist.Field(name='u', bases=(grid.xbasis, grid.ybasis))
    v = grid.dist.Field(name='v', bases=(grid.xbasis, grid.ybasis))

    # initial condition
    u['g'] = 0.1*random_normal(grid_size=grid.grid_size, random_state=random_state)
    v['g'] = 0.1*random_normal(grid_size=grid.grid_size, random_state=random_state)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, grid.coords['x'])
    dy = lambda A: d3.Differentiate(A, grid.coords['y'])

    # Problem
    problem = d3.IVP([u, v], namespace=locals())

    # Add main equation, with linear terms on the LHS and nonlinear terms on the RHS
    # first part:
    LHS = f"{1/rate}*dt(u) - {diffusivity_1}*lap(u) - {reaction_speed_1}*(u - v)"
    RHS = f"-{reaction_speed_1} * (u**3 - {k})"
    problem.add_equation(LHS+"="+RHS)

    # second part:
    LHS = f"{1/rate}*dt(v) - {diffusivity_2}*lap(v) - {reaction_speed_2}*(u - v)"
    RHS = f"0"
    problem.add_equation(LHS+"="+RHS)

    return problem

def build_problem_wave(grid, 
                       wave_speed: float=2.1,
                       rate:float=1, 
                       random_state=None):

    # Fields
    u = grid.dist.Field(name='u', bases=(grid.xbasis, grid.ybasis))
    du = grid.dist.Field(name='du', bases=(grid.xbasis, grid.ybasis))

    # initial condition
    u['g'] = sum_of_gaussians(grid_size=grid.grid_size, components=5, zero_level=2, random_state=random_state)
    du['g'] = np.zeros(grid.grid_size)

    # Substitutions
    dx = lambda A: d3.Differentiate(A, grid.coords['x'])
    dy = lambda A: d3.Differentiate(A, grid.coords['y'])

    # Problem
    problem = d3.IVP([u, du], namespace=locals())

    # Add main equation, with linear terms on the LHS and nonlinear terms on the RHS
    LHS = f"{1/rate}*dt(du) - {wave_speed**2}*lap(u)"
    RHS = "0"
    problem.add_equation(LHS+"="+RHS)

    # 
    LHS = f"{1/rate}*dt(u) - du"
    RHS = "0"
    problem.add_equation(LHS+"="+RHS)

    return problem