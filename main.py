from ortools.linear_solver import pywraplp

solver = pywraplp.Solver(name="Test Solver", problem_type=pywraplp.Solver.SAT_INTEGER_PROGRAMMING)

infinity = solver.infinity()

x = solver.IntVar(0, infinity, "x")
y = solver.IntVar(0, infinity, "y")

solver.Add(x >= y)

solver.Maximize(-x + -y)

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
    print("Solution:")
    print("Objective value =", solver.Objective().Value())
    print("x =", x.solution_value())
    print("y =", y.solution_value())
else:
    print("The problem does not have an optimal solution.")