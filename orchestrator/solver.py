from typing import *
from ortools.linear_solver import pywraplp as lp

from orchestrator import VarTable

D = TypeVar('D')
J = TypeVar('J')

def make_qtable(solver: lp.Solver, jobs: Iterable[J], devices: Iterable[D], time: Iterable[int]) -> VarTable:
	"""
	:param jobs: An iterable of the types of jobs
	:param devices: An iterable of all the devices
	:param time: An iterable of all the timestamps over which to make these variables
	:return: A table of variables that represent "number of `job` that should be completed on `device` before `time`"
	"""
	table: VarTable = VarTable(
		solver, "complete", job=set(jobs), device=set(devices), time=set(time)
	)
	return table

def make_mtable(solver: lp.Solver, jobs: Iterable[J], devices: Iterable[D], time: Iterable[int]) -> VarTable:
	"""
	:return: A table of variables that represent "The number of `job` outputs that have moved from `src` to `dst` before `time`"
	"""
	device_set = set(devices)
	table: VarTable = VarTable(
		solver, "move", job=set(jobs), src=device_set, dst=device_set, time=set(time)
	)
	return table

def multithreading_constraint(
	qtable: VarTable, mtable: VarTable, devices: Iterable[D], jobs: Iterable[J], device: D, time: Iterable[int]
):
	