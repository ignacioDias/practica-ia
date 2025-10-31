import random
from itertools import permutations
from graph import Graph

NODES = ['A', 'B', 'C', 'D', 'E', 'F', 'G']
POP_SIZE = 50
ITERATIONS = 40
CXPB = 0.8
MUTPB = 0.1

# === Grafo con costos y tiempos ===
g = Graph(NODES)
edges = [
    ('A', 'B', 10, 20),
    ('A', 'C', 15, 22),
    ('A', 'D', 20, 25),
    ('B', 'C', 12, 17),
    ('B', 'D', 18, 19),
    ('C', 'E', 19, 30),
    ('D', 'E', 16, 12),
    ('E', 'F', 21, 35),
    ('F', 'G', 17, 10),
    ('E', 'G', 23, 15),
]
for (a, b, c, t) in edges:
    g.add_path(a, b, c, t)

g.show()

def generate_individual():
    return random.sample(NODES, len(NODES))

def generate_population():
    return [generate_individual() for _ in range(POP_SIZE)]

def fitness(individual):
    cost, time = g.calculate_cost_and_time(individual)
    return (cost, time)

def dominates(f1, f2):
    """Devuelve True si f1 domina a f2 (mejor en ambos objetivos)."""
    return (f1[0] <= f2[0] and f1[1] <= f2[1]) and (f1 != f2)

def pareto_front(population):
    front = []
    for i, ind1 in enumerate(population):
        dominated = False
        for j, ind2 in enumerate(population):
            if i != j and dominates(fitness(ind2), fitness(ind1)):
                dominated = True
                break
        if not dominated:
            front.append(ind1)
    return front

def crossover(p1, p2):
    n = len(p1)
    i, j = sorted(random.sample(range(n), 2))
    child1 = [None] * n
    child1[i:j] = p1[i:j]
    p2_items = [g for g in p2 if g not in child1]
    k = 0
    for idx in list(range(0, i)) + list(range(j, n)):
        child1[idx] = p2_items[k]
        k += 1
    return child1

def mutate(ind):
    if random.random() < MUTPB:
        i, j = random.sample(range(len(ind)), 2)
        ind[i], ind[j] = ind[j], ind[i]
    return ind

def selection(pop):
    k = 3
    selected = []
    for _ in range(POP_SIZE):
        aspirants = random.sample(pop, k)
        best = min(aspirants, key=lambda ind: sum(fitness(ind))) 
        selected.append(best)
    return selected

def moga():
    population = generate_population()
    for gen in range(ITERATIONS):
        pareto = pareto_front(population)
        print(f"Generación {gen} — Frente de Pareto: {len(pareto)} individuos")
        next_gen = []
        selected = selection(population)
        for i in range(0, POP_SIZE, 2):
            if random.random() < CXPB:
                c1 = crossover(selected[i], selected[i + 1])
                c2 = crossover(selected[i + 1], selected[i])
            else:
                c1, c2 = selected[i], selected[i + 1]
            next_gen.extend([mutate(c1), mutate(c2)])
        population = next_gen
    return pareto_front(population)

pareto = moga()
print("\n🌈 Frente de Pareto final:")
for ind in pareto:
    cost, time = fitness(ind)
    print(f"{ind} → Cost={cost}, Time={time}")
