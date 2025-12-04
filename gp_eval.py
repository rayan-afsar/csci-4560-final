from deap import gp, base
import random
import csv
import numpy as np
import math
from Game2048 import Game2048, extract_features, UP, DOWN, LEFT, RIGHT
from gp_2048 import run_game


# initialize primitive set (potential nodes for GP tree)
pset = gp.PrimitiveSet("MAIN", 9)   # 6 input features
pset.renameArguments(ARG0="empty", ARG1="max_tile", ARG2="sum_tiles", ARG3="smooth", ARG4="variance", ARG5="monotonic", ARG6="edge_weight", ARG7="max_corner", ARG8="potential_merges")


def add(a, b):
    return float(a + b)
def sub(a, b):
    return float(a - b)
def mul(a, b):
    return float(a * b)

# safe arithmetic (no overflow) primitives
def safe_div(a, b):
    return float(a / b) if abs(b) > 1e-9 else a
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
    return float(abs(a))
def ceil(a):
    return float(math.ceil(a))
def floor(a):
    return float(math.floor(a))
def hypot(a, b):
    return float(math.hypot(a, b)) # hypotenuse of triangle w/ sides a & b

# boolean primitives
def logical_and(a, b): 
    return 1 if (a and b) else 0
def logical_or(a, b): 
    return 1 if (a or b) else 0
def logical_not(a): 
    return 1 if not a else 0

pset.addPrimitive(add, 2)
pset.addPrimitive(sub, 2)
pset.addPrimitive(mul, 2)

pset.addPrimitive(safe_div, 2)
pset.addPrimitive(safe_log, 1)
pset.addPrimitive(safe_sqrt, 1)

pset.addPrimitive(gt, 2)
pset.addPrimitive(lt, 2)
pset.addPrimitive(eq, 2)

pset.addPrimitive(max2, 2)
pset.addPrimitive(min2, 2)
pset.addPrimitive(abs1, 1)
pset.addPrimitive(ceil, 1)
pset.addPrimitive(floor, 1)
pset.addPrimitive(hypot, 2)

pset.addPrimitive(logical_and, 2)
pset.addPrimitive(logical_or, 2)
pset.addPrimitive(logical_not, 1)

# terminals (final move)
pset.addTerminal(0, name="up")  # up
pset.addTerminal(1, name="down")  # down
pset.addTerminal(2, name="left")  # left
pset.addTerminal(3, name="right")  # right

# ephemeral constant to add random noise w/o making python freak out
def rand_const():
    return random.random()

pset.addEphemeralConstant("rand", rand_const)

best_str = "mul(monotonic, max2(potential_merges, mul(add(floor(empty), add(potential_merges, max2(logical_or(max_corner, max_corner), hypot(edge_weight, logical_or(hypot(floor(floor(left)), add(lt(edge_weight, floor(smooth)), empty)), mul(max_corner, mul(add(floor(add(potential_merges, empty)), add(potential_merges, empty)), edge_weight))))))), edge_weight)))"
best_ind = gp.PrimitiveTree.from_string(best_str, pset)

toolbox = base.Toolbox()
toolbox.register("compile", gp.compile, pset=pset)
expr = toolbox.compile(expr=best_ind)

# test best individual at end over 50 games
results = []
for i in range(100):
    mx, sc, mv = run_game(expr, seed=random.randrange(1,10000))
    results.append((i, mx, sc, mv))
    print("Game " + str(i) + " complete")

avg_max = np.mean([r[1] for r in results])
avg_score = np.mean([r[2] for r in results])
avg_moves = np.mean([r[3] for r in results])

csv_path = "_best_individual_eval3.csv"

print("Results over 100 evals:")
with open(csv_path, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Game", "MaxTile", "Score", "Moves"])
    writer.writerows(results)
    writer.writerow([])
    writer.writerow(["Averages", avg_max, avg_score, avg_moves])

print(f"Saved evaluation results to {csv_path}")






