import random
import csv
import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
from Game2048 import Game2048, extract_features, UP, DOWN, LEFT, RIGHT
from gp_2048 import run_game
from scoop import futures
from functools import partial

N_FEATURES = 9

# bounds for weights 
BOUND_LOW = -10.0
BOUND_HIGH = 10.0

def expr_from_weights(weights):
    def expr(*feats):
        return float(np.dot(weights, feats))
    return expr

def initES(ind_class, strat_class):
    """
    Creates an individual whose .strategy is a vector of sigmas.
    """
    ind = ind_class([random.gauss(0, 1) for _ in range(N_FEATURES)])
    ind.strategy = strat_class([1.0] * N_FEATURES)   # sigma for each gene
    return ind

creator.create("FitnessMax2048", base.Fitness, weights=(1.0,))
creator.create("Controller", list, fitness=creator.FitnessMax2048, strategy=None)
creator.create("Strategy", list)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)
toolbox.register("attr_float", np.random.normal, 0, 1)
toolbox.register("individual", 
                 partial(initES, creator.Controller, creator.Strategy))
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

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

toolbox.register("evaluate", evaluate)
toolbox.register("mate", tools.cxESBlend, alpha=0.5)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.2)
toolbox.register("select", tools.selTournament, tournsize=3)

def main():
    MU = 25
    LAMBDA = 150
    NGEN = 200

    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(3)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std dev", np.std)
    stats.register("min score", np.min)
    stats.register("max score", np.max)

    pop, log = algorithms.eaMuCommaLambda(
        population=pop,
        toolbox=toolbox,
        mu=MU,
        lambda_=LAMBDA,
        cxpb=0.2,
        mutpb=0.8,
        ngen=NGEN,
        stats=stats,
        halloffame=hof,
        verbose=True
    )

    log_df = pd.DataFrame(log)
    log_name = "_es3_log.csv"
    log_df.to_csv(log_name, index=False)
    print("Saved evolution log to " + log_name + "\n")

    best_weights = hof[0]
    best_expr = expr_from_weights(best_weights)
    print("Best Individual: " + str(best_weights))

    results = []
    for i in range(100):
        mx, sc, mv = run_game(best_expr, seed=random.randrange(1,1000))
        results.append((i, mx, sc, mv))
        print("Game " + str(i) + " complete")

    avg_max = np.mean([r[1] for r in results])
    avg_score = np.mean([r[2] for r in results])
    avg_moves = np.mean([r[3] for r in results])

    csv_path = "_best_es_eval3.csv"

    print("Results over 100 evals:")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Game", "MaxTile", "Score", "Moves"])
        writer.writerows(results)
        writer.writerow([])
        writer.writerow(["Averages", avg_max, avg_score, avg_moves])

    print(f"Saved evaluation results to {csv_path}")

    return pop, log, hof

if __name__ == "__main__":
    main()