import random
import math
import operator
from scoop import futures
import numpy as np
from deap import base, creator, gp, tools, algorithms
from Game2048 import Game2048, extract_features, UP, DOWN, LEFT, RIGHT

# initialize primitive set (potential nodes for GP tree)
pset = gp.PrimitiveSet("MAIN", 9)   # 6 input features
pset.renameArguments(ARG0="empty", ARG1="max_tile", ARG2="sum_tiles", ARG3="smooth", ARG4="variance", ARG5="monotonic", ARG6="edge_weight", ARG7="max_corner", ARG8="potential_merges")

# safe arithmetic (no overflow) primitives
def safe_div(a, b):
    return a / b if abs(b) > 1e-9 else a
def safe_log(a):
    return math.log(abs(a)+1)
def safe_sqrt(a):
    return math.sqrt(abs(a))

# comparison primitives
def gt(a, b): 
    return 1 if a > b else 0
def lt(a, b): 
    return 1 if a < b else 0
def eq(a, b): 
    return 1 if a == b else 0

# return max/min/abs
def max2(a, b): 
    return a if a > b else b
def min2(a, b): 
    return a if a < b else b
def abs1(a):
    return abs(a)

# boolean primitives
def logical_and(a, b): 
    return 1 if (a and b) else 0
def logical_or(a, b): 
    return 1 if (a or b) else 0
def logical_not(a): 
    return 1 if not a else 0


pset.addPrimitive(operator.add, 2)
pset.addPrimitive(operator.sub, 2)
pset.addPrimitive(operator.mul, 2)

pset.addPrimitive(safe_div, 2)
pset.addPrimitive(safe_log, 1)
pset.addPrimitive(safe_sqrt, 1)

pset.addPrimitive(gt, 2)
pset.addPrimitive(lt, 2)
pset.addPrimitive(eq, 2)

pset.addPrimitive(max2, 2)
pset.addPrimitive(min2, 2)
pset.addPrimitive(abs1, 1)

pset.addPrimitive(logical_and, 2)
pset.addPrimitive(logical_or, 2)
pset.addPrimitive(logical_not, 1)

# terminals (final move)
pset.addTerminal(0, name="up")  # up
pset.addTerminal(1, name="down")  # down
pset.addTerminal(2, name="left")  # left
pset.addTerminal(3, name="right")  # right

pset.addEphemeralConstant("rand", random.random)

# setting up GP w/ DEAP
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("map", futures.map)
toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# eval of a GP individual
def run_game(expr, seed=None, max_moves=3000):
    rng=random.Random(seed)
    game = Game2048(rng=rng)
    moves = 0
    while game.can_move() and moves < max_moves:
        scores = []
        for move in [UP, DOWN, LEFT, RIGHT]:
            g2 = Game2048()
            g2.board = game.board.copy()
            g2.score = game.score
            if not g2.move(move):
                scores.append(-1e9)
                continue
            feats = extract_features(g2.board)
            empty_bonus = feats[0] * 3    # extra weight for empty tiles
            monotonicity = feats[3] * 1.5
            val = expr(*feats) + empty_bonus + monotonicity
            scores.append(val)
        
        best_move = np.argmax(scores)
        game.move(best_move)
        moves += 1

    return game.get_max_tile(), game.score, moves

def evaluate_individual(individual):
    expr = toolbox.compile(expr=individual)
    max_tiles = []
    moves_list = []
    for i in range(3):
        mx, sc, mv = run_game(expr, seed=random.randrange(1, 100))
        max_tiles.append(mx)
        moves_list.append(mv)

    avg_max = np.mean(max_tiles)
    bonus = 1000 if any(t >= 2048 for t in max_tiles) else 0
    return (avg_max + bonus,)

# register GP functions
toolbox.register("compile", gp.compile, pset=pset)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("mate", gp.cxOnePoint)
toolbox.register("expr_mut", gp.genHalfAndHalf, min_=1, max_=3)
toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

toolbox.decorate("mate", gp.staticLimit(key=len, max_value=40))
toolbox.decorate("mutate", gp.staticLimit(key=len, max_value=40))

# main loop
def main():
    pop = toolbox.population(n=1000)
    hof = tools.HallOfFame(5)

    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg score", np.mean)
    stats.register("max score", np.max)

    pop, log = algorithms.eaSimple(
        pop, toolbox, 
        cxpb=0.7, mutpb=0.1, 
        ngen=100, 
        stats=stats, 
        halloffame=hof,
        verbose=True
    )

    print("\nBest Individual:")
    print(hof[0])
    expr = toolbox.compile(expr=hof[0])

    # test best individual at end over 30 games
    results = []
    for _ in range(40):
        mx, sc, mv = run_game(expr, seed=random.randrange(1,1000))
        results.append((mx, sc, mv))

    print("Results over 40 evals:")
    for idx, (mx, sc, mv) in enumerate(results):
        print(f"Game {idx+1}: Max tile={mx}, Score={sc}, Moves={mv}")

    avg_max = np.mean([r[0] for r in results])
    avg_score = np.mean([r[1] for r in results])
    avg_moves = np.mean([r[2] for r in results])
    print(f"\nAverages: Max tile={avg_max:.1f}, Score={avg_score:.1f}, Moves={avg_moves:.1f}")
    return pop, log, hof


if __name__ == "__main__":
    main()
