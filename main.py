from typing import List, TypeVar, Tuple
from ortools.linear_solver import pywraplp


class Device:
    def getThreads(self) -> int:
        pass


D = TypeVar('D', bound=Device)
J = TypeVar('J')


class EnvironmentStatistics:
    def getExecutionTime(self, device: D, job: J) -> int:
        pass

    def getEncodingTime(self, source: D, destination: D, job: J) -> int:
        pass

    def getTransitTime(self, source: D, destination: D, job: J) -> int:
        pass


def orchestrate(statistics: EnvironmentStatistics, devices: List[D], jobs: List[J], dependencies: List[Tuple[J, J, int]],
                timesteps: int):
    solver = pywraplp.Solver.CreateSolver("SAT")
    if not solver:
        raise AssertionError("No SAT solver found!")

    infinity = solver.infinity()

    q = []
    m = []
    for j in range(len(jobs)):
        qj = []
        for d in range(len(devices)):
            qjd = []
            for t in range(timesteps):
                qjdt = solver.IntVar(0.0, infinity, f"q{j}{d}{t}")
                qjd.append(qjdt)
            qj.append(qjd)
        q.append(qj)

        mj = []
        for d in range(len(devices)):
            mjd = []
            for d2 in range(len(devices)):
                mjdd2 = []
                for t in range(timesteps):
                    mjdd2t = solver.IntVar(0.0, infinity, f"m{j}{d}{d2}{t}")
                    mjdd2.append(mjdd2t)
                mjd.append(mjdd2)
            mj.append(mjd)
        m.append(mj)

    for t in range(timesteps):
        for di, d in enumerate(devices):
            # A computer can only perform as many operations at a time as it has threads.
            q_sum_conc = 0
            m_sum_conc = 0
            for ji, j in enumerate(jobs):
                q_sum_conc += q[ji][di][t + statistics.getExecutionTime(d, j)]
                q_sum_conc -= q[ji][di][t]

                m_sum_trans = 0
                for d2i, d2 in enumerate(devices):
                    m_sum_conc += m[ji][di][d2i][t + statistics.getEncodingTime(d, d2, j)]
                    m_sum_conc -= m[ji][di][d2i][t]
                    m_sum_trans += m[ji][di][d2i][t
                                                  + statistics.getExecutionTime(j, d)
                                                  + statistics.getEncodingTime(d, d2, j)
                                                  + statistics.getTransitTime(d, d2, j)
                                                  ]
                temp = q[ji][di][t]
                solver.Add(temp >= m_sum_trans)
            solver.Add(q_sum_conc + m_sum_conc <= d.getThreads())

            for ri, r in enumerate(dependencies):
                a, b, count = r
                m_sum_deps = 0
                for srci, src in enumerate(devices):
                    m_sum_deps += count * m[jobs.index(a)][srci][di][t]
                bi = jobs.index(b)
                solver.Add(m_sum_deps >= q[bi][di][t + statistics.getExecutionTime(bi)])
