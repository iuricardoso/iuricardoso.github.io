# -*- coding: utf-8 -*-

'''
    CSP - Jobshop OR-Tools
'''

from ortools.sat.python import cp_model
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Dados do problema: cada job tem 5 tarefas (máquina, duração)
jobs_data = [
    [(1, 72), (0, 87), (4, 95), (2, 66), (3, 60)],
    [(4, 5), (3, 35), (0, 48), (2, 39), (1, 54)],
    [(1, 46), (3, 20), (2, 21), (0, 97), (4, 55)],
    [(0, 59), (3, 19), (4, 46), (1, 34), (2, 37)],
    [(4, 23), (2, 73), (3, 25), (1, 24), (0, 28)],
    [(3, 28), (0, 45), (4, 5), (1, 78), (2, 83)],
    [(0, 53), (3, 71), (1, 37), (4, 29), (2, 12)],
    [(4, 12), (2, 87), (3, 33), (1, 55), (0, 38)],
    [(2, 49), (3, 83), (1, 40), (0, 48), (4, 7)],
    [(2, 65), (3, 17), (0, 90), (4, 27), (1, 23)],
]


def main():
    # Modelo CSP
    model = cp_model.CpModel()
    num_jobs = len(jobs_data)
    num_tasks = len(jobs_data[0])
    all_jobs = range(num_jobs)

    # Máquinas e tempo de operação
    machines = set(m for job in jobs_data for m, _ in job)
    time = sum(d for job in jobs_data for _, d in job)

    # Dicionários de tarefas e intervalos por máquina
    all_tasks = {}
    machine_intervals = {m: [] for m in machines}
    makespan_tasks = []

    for job_id in all_jobs:
        for task_id, (machine, duration) in enumerate(jobs_data[job_id]):
            suffix = f"_{job_id}_{task_id}"
            start = model.NewIntVar(0, time, "start" + suffix)
            end = model.NewIntVar(0, time, "end" + suffix)
            interval = model.NewIntervalVar(start, duration, end, "interval" + suffix)
            all_tasks[(job_id, task_id)] = (start, end, interval, machine, duration)
            machine_intervals[machine].append(interval)
        makespan_tasks.append(all_tasks[(job_id, num_tasks - 1)][1])

    # Restrição: precedência das tarefas em cada job
    for job_id in all_jobs:
        for task_id in range(num_tasks - 1):
            model.Add(
                all_tasks[(job_id, task_id + 1)][0] >= all_tasks[(job_id, task_id)][1]
            )

    # Restrição: tarefas não sobrepõem na mesma máquina
    for m in machines:
        model.AddNoOverlap(machine_intervals[m])

    # Variável makespan e objetivo (minimizar tempo)
    makespan = model.NewIntVar(0, time, "makespan")
    model.AddMaxEquality(makespan, makespan_tasks)
    model.Minimize(makespan)

    # Resolução do modelo com solver CP-SAT
    solver = cp_model.CpSolver()
    status = solver.Solve(model)

    # Impressão do Resultado + Gráfico de Gantt
    if status in [cp_model.OPTIMAL, cp_model.FEASIBLE]:
        print(f"\nMakespan: {solver.Value(makespan)} horas\n")
        fig, ax = plt.subplots(figsize=(15, 6))
        colors = plt.cm.get_cmap("tab20", num_jobs)

        for job_id in all_jobs:
            print(f"Job {job_id + 1}:")
            for task_id, (machine, duration) in enumerate(jobs_data[job_id]):
                start = solver.Value(all_tasks[(job_id, task_id)][0])
                print(f"  Task {task_id + 1} (Machine {machine}, {duration}h): Start at {start}")

                # Gantt
                ax.barh(
                    y=machine,
                    width=duration,
                    left=start,
                    height=0.6,
                    color=colors(job_id),
                    edgecolor="black",
                )
                ax.text(
                    start + duration / 2,
                    machine,
                    f"J{job_id+1}-T{task_id+1}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )
            print()

        ax.set_xlabel("Tempo (horas)")
        ax.set_ylabel("Máquina")
        ax.set_yticks(sorted(machines))
        ax.set_yticklabels([f"Máquina {m}" for m in sorted(machines)])
        ax.set_title("Job Shop Scheduling")
        plt.grid(True, axis="x", linestyle="--", alpha=0.4)
        plt.tight_layout()
        plt.show()
    else:
        print("Nenhuma solução encontrada.")


if __name__ == "__main__":
    main()
