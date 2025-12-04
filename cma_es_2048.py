import random
import numpy as np
import csv
from deap import base, creator, tools, cma
from scoop import futures
from gp_2048 import run_game 

N_FEATURES = 9

def expr_from_weights(weights):
    def expr(*feats):
        return float(np.dot(weights, feats))
    return expr

creator.create("FitnessMax2048CMA", base.Fitness, weights=(1.0,))
creator.create("IndividualCMA", list, fitness=creator.FitnessMax2048CMA)

def evaluate(individual):
    expr = expr_from_weights(individual)
    max_tiles = []
    moves_list = []
    scores = []
    for i in range(3):
        mx, sc, mv = run_game(expr, seed=random.randrange(1, 10000))
        max_tiles.append(mx)
        moves_list.append(mv)
        scores.append(sc)

    avg_score = np.mean(scores)
    avg_max = np.mean(max_tiles)
    bonus = 1000 if any(t >= 2048 for t in max_tiles) else 0
    return (avg_score + avg_max + bonus,)

strategy = cma.Strategy(
    centroid=[0.0] * N_FEATURES,    # start point of evolution
    sigma = 1.5,    # initial step size
    lambda_ = 200,   # pop size
    mu = 100     # parents for next gen
)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)
toolbox.register("generate", strategy.generate, creator.IndividualCMA)
toolbox.register("update", strategy.update)
toolbox.register("evaluate", evaluate)

def main():
    LOG_CSV = "cma_es_log5.csv"
    with open(LOG_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "generation",
            "mean_fitness",
            "std_dev",
            "worst_fitness",
            "best_fitness",
            "sigma",
        ])

    NGEN = 75
    hof = tools.HallOfFame(3)

    for gen in range(NGEN):
        population = toolbox.generate()

        fitnesses = list(map(toolbox.evaluate, population)) # eval
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        toolbox.update(population)  # update cma strat
        hof.update(population)

        best_fit = max(ind.fitness.values[0] for ind in population)
        worst_fit = min(ind.fitness.values[0] for ind in population)
        mean_fit = np.mean([ind.fitness.values[0] for ind in population])
        std_dev = np.std([ind.fitness.values[0] for ind in population])
        sigma = strategy.sigma

        print(f"Gen {gen}: mean={mean_fit:.2f}, std dev={std_dev:.2f}, worst={worst_fit:.2f}, best={best_fit:.2f},  sigma={sigma:.4f}")

        # Save to CSV
        with open(LOG_CSV, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([gen, mean_fit, std_dev, worst_fit, best_fit, sigma])

    np.savetxt("best_cma_es_weights5.txt", hof[0])

    best_w = hof[0]
    best_expr = expr_from_weights(best_w)
    results = []
    for i in range(100):
        mx, sc, mv = run_game(best_expr, seed=random.randrange(1,1000))
        results.append((i, mx, sc, mv))
        print("Game " + str(i) + " complete")

    avg_max = np.mean([r[1] for r in results])
    avg_score = np.mean([r[2] for r in results])
    avg_moves = np.mean([r[3] for r in results])

    csv_path = "_best_cma_es_eval5.csv"

    print("Results over 100 evals:")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Game", "MaxTile", "Score", "Moves"])
        writer.writerows(results)
        writer.writerow([])
        writer.writerow(["Averages", avg_max, avg_score, avg_moves])

    print(f"Saved evaluation results to {csv_path}")


if __name__ == "__main__":
    main()