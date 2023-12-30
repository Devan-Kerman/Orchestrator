from typing import *
import numpy as np
import itertools
from ortools.linear_solver import pywraplp as lp

class VarTable(object):
	"""
	Dense storage for constraint variables based on lookup
	"""
	solver: lp.Solver
	key_map: Dict[str, Dict[Any, int]] = {}
	dense: np.ndarray

	def __init__(self, solver: lp.Solver, name: str, lower=0, upper=lp.Solver.infinity(), **axis: set):
		self.solver = solver
		size = tuple()
		for k, v in axis.items():
			self.key_map[k] = {v: i for i, v in enumerate(v)}
			size += (len(v),)
		self.dense = np.empty(shape=size, dtype=np.object_)

		for place in itertools.product(*axis.values()):
			indices = tuple(self.key_map[k][v] for k, v in zip(self.key_map.keys(), place))
			variable = solver.IntVar(lower, upper, f"{name}{place}")
			self.dense[indices] = variable

	def find(self, **kwargs):
		"""
		Returns none if any of the values are not found in the keymap
		"""
		indices = tuple(
			self.key_map[k][v] if v in self.key_map[k] else None for k, v in kwargs.items()
		)
		if None in indices:
			return None
		else:
			return self.dense[indices]

	def __getitem__(self, item):
		indices = tuple(self.key_map[k][v] for k, v in zip(self.key_map.keys(), item))
		return self.dense[indices]

	def __contains__(self, item):
		indices = tuple(self.key_map[k][v] if v in self.key_map[k] else None for k, v in zip(self.key_map.keys(), item))
		return indices in self.dense
