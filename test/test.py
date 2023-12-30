import numpy as np
from ortools.linear_solver import pywraplp as lp
from orchestrator import VarTable

arr = np.array([
    [1, 0],
    [0, 1],
])
print(arr)
print(arr.dtype)
print(arr[tuple([0, 0])])

solver = lp.Solver(name="Test Solver", problem_type=lp.Solver.SAT_INTEGER_PROGRAMMING)
table = VarTable(solver, "test",
                 x={1, 2, 3, 4},
                 y={2, 4, 5, 6},
                 z={3, 1, 4, 5}
                 )

print(table.find(x=1, y=2, z=1))
print(table[1, 2, 1])