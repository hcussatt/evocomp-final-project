"""
tetris_classical_evo.py: basic loop for playing NES Tetris one piece at a time.

Put this file in the same folder as nes_tetris.py and run it.
Edit the SETTINGS section and the choose_moves() function to experiment.
"""

import random
import copy
from pathlib import Path
from tempfile import NamedTemporaryFile
import numpy as np
from nes_tetris import NESTetris, ORIENTATIONS

SEED = 67
rng = np.random.default_rng(67)

# Housekeeping - be able to see where wells are, where holes are, height of the cols, etc.
def analyze_board(board):
    rows = len(board)
    cols = len(board[0])

    heights = []
    holes = []

    for col in range(cols):
        seen_block = False
        height = 0

        for row in range(rows):
            if board[row][col] != 0:
                if not seen_block:
                    height = rows - row
                seen_block = True

            elif seen_block:
                holes.append((row, col))

        heights.append(height)

    wells = []

    for col in range(cols):
        left = heights[col - 1] if col > 0 else rows
        right = heights[col + 1] if col < cols - 1 else rows

        depth = max(0, min(left, right) - heights[col])
        wells.append(depth)

    deepest_col = max(range(cols), key=lambda c: wells[c])

    bumpiness = sum(
        abs(heights[i] - heights[i + 1])
        for i in range(cols - 1)
    )

    return {
        "column_heights": heights,
        "holes": holes,
        "hole_count": len(holes),
        "wells": wells,
        "deepest_well": (deepest_col, wells[deepest_col]),
        "bumpiness": bumpiness,
        "max_height": max(heights),
    }

def get_bell_curve_number(center, low=0, high=9, scale=1.5):
  """Generates a random number from a normal (bell curve) distribution

    centered around 'center' and clamped/rounded to [low, high].
    """
  # Draw from a normal distribution
  val = np.random.normal(loc=center, scale=scale)

  # Round to the nearest integer
  val_int = int(round(val))

  # Clamp between low and high (0 through 9)
  return np.clip(val_int, low, high)

# ----------------------------------------------------------------------------
# MUTATION LOGIC - Currently set to be totally random, but has capability to enter target_column decisions 
# ----------------------------------------------------------------------------
"""Replacement for choose_moves() in tetris_classical_evo.py.

Replace the old function with the function below. Keep the existing np, rng,
ORIENTATIONS, and analyze_board definitions/imports in your main program.
Call choose_moves(game, strategy), passing a game object, not its state dict.

choices=3 retains exploration among a few good candidates. Set choices=1
to select the best-ranked verified candidate. No changes to EA fitness or
generation limits are made by this replacement.
"""


def choose_moves(game, strategy, choices=3):
    """Return (rotation, pivot_column), or None if no candidate verifies.

    dig: favor a flatter surface; use line clears to break ties. Ignore holes.
    wells: minimize added holes, then favor filling deeper existing wells.
    height: retain support for minimizing average height.

    Call at the start of a piece, before sending movement inputs. Candidates
    use the wrapper's rotate/slide/drop model and are checked on one reusable
    emulator clone. The original game is never advanced by this function.
    """
    if strategy not in ("dig", "wells", "height"):
        raise ValueError(f"Unknown strategy: {strategy}")
    if not isinstance(choices, int) or choices < 1:
        raise ValueError("choices must be a positive integer")
    if game.game_over or game.current_piece is None:
        return None

    board = game.board()
    before = analyze_board(board)
    heights = before["column_heights"]
    wells = before["wells"]

    # Guard against the wrapper's missing initial-overlap check.
    if not game._fits(board, game.orientation_id, *game.piece_xy):
        return None

    groups = {}

    for rotation, column in game.legal_placements():
        simulated = game.simulate_placement(rotation, column)
        if simulated is None:
            continue
        after_board, lines, topped_out = simulated
        if topped_out:
            continue

        after = analyze_board(after_board)
        flatness = float(np.var(after["column_heights"]))

        if strategy == "wells":
            landing = game._resolve(rotation, column, board)
            if landing is None:
                continue
            orientation, x, y = landing
            cells = {
                (y + dy, x + dx)
                for dy, dx in ORIENTATIONS[orientation][1]
            }

            # Count only bottom-up filling of a well. Bridging its opening
            # receives no well-filling credit. Use cells BEFORE line clears.
            deepest_filled = 0
            well_cells_filled = 0
            for col, depth in enumerate(wells):
                filled = 0
                bottom = 19 - heights[col]
                while filled < depth and (bottom - filled, col) in cells:
                    filled += 1
                if filled:
                    deepest_filled = max(deepest_filled, depth)
                    well_cells_filled += filled

            # Zero added holes outranks ANY placement that adds holes.
            # If none verifies, the next-smallest hole increase is allowed.
            added_holes = max(0, after["hole_count"] - before["hole_count"])
            priority = (added_holes, -deepest_filled)
            rank = (-well_cells_filled, flatness, -lines)
        elif strategy == "dig":
            # No protected column and no hole rejection or hole penalty.
            priority = (0, 0)
            rank = (flatness, -lines)
        else:  # Existing optional "height" strategy.
            priority = (0, 0)
            rank = (float(np.mean(after["column_heights"])), -lines)

        groups.setdefault(priority, []).append(
            (rank, rotation, column, after_board, lines)
        )

    if not groups:
        return None

    trial = game.clone()
    try:
        start = trial.snapshot()
        for priority in sorted(groups):
            candidates = groups[priority]
            # Randomize equal ranks to avoid a permanent left/right bias.
            candidates = [candidates[i] for i in rng.permutation(len(candidates))]
            candidates.sort(key=lambda candidate: candidate[0])

            # Try a few good alternatives, then lower-ranked batches if
            # their actual button execution fails to match the prediction.
            for offset in range(0, len(candidates), choices):
                batch = candidates[offset:offset + choices]
                for index in rng.permutation(len(batch)):
                    _, rotation, column, expected_board, expected_lines = batch[index]
                    trial.restore(start)
                    result = trial.place(rotation, column)
                    if (
                        not trial.game_over
                        and result["lines_cleared"] == expected_lines
                        and np.array_equal(trial.board(), expected_board)
                    ):
                        return rotation, column
    finally:
        trial.close()

    return None



# ----------------------------------------------------------------------------
# MAIN GAME - UNUSED (LOGIC IMPLEMENTED IN EVOLUTIONARY LOOP)
# ----------------------------------------------------------------------------
# def play_game(num_pieces=NUM_PIECES, seed=SEED, show_board=SHOW_BOARD):
#     random.seed(seed)
#     game = NESTetris(seed=seed)

#     for turn in range(1, num_pieces + 1):
#         if game.game_over:
#             print(f"Game over after {turn - 1} pieces.")
#             break

#         # 1) read the game state
#         state = game.get_state()

#         # 2) decide what to do with this piece
#         moves = choose_moves(state)

#         # 3) send the moves to the emulator (the piece drops and locks)
#         result = game.play_moves(moves)

#         # 4) read the new state
#         heights_after = game.column_heights().tolist()
#         print(f"Turn {turn:3d} | piece {state['current']} | moves {moves}")
#         print(f"         heights {heights_after} | lines cleared {result['lines_cleared']}")
#         if show_board:
#             print(game.render_text())
#             print()

#     print("\nFinal:", {"lines": game.lines, "score": game.score,
#                        "pieces": game.pieces_placed, "game_over": game.game_over})
#     return game


# ----------------------------------------------------------------------------
# EVOLUTIONARY ASPECT
# ----------------------------------------------------------------------------
BLOCK_PIECES = 10
PIECES = ("I", "O", "T", "S", "Z", "J", "L")
GOALS = ("flatness", "lowness")
POLICY_PATH = Path(__file__).with_name("tetris_pareto_policies.npz")


def policy_inputs(state):
    features = analyze_board(state["board"])
    heights = features["column_heights"]
    # Bias, mean height, spread in heights, deepest well and buried cells.
    return [1.0, np.mean(heights) / 20, np.std(heights) / 10,
            features["deepest_well"][1] / 20, features["hole_count"] / 200]


def pareto_order(population, goal=None):
    """Order viable candidates by non-dominated front and crowding distance."""
    remaining = [p for p in population if all(np.isfinite(p.fitness))]
    ordered = []
    rank = 0
    while remaining:
        front = []
        for p in remaining:
            flat, low = p.fitness
            if not any(q.fitness[0] <= flat and q.fitness[1] <= low
                       and q.fitness != p.fitness for q in remaining):
                front.append(p)
        # Measure spacing on distinct objective pairs so duplicates get no
        # extra credit for appearing at the edge of a front.
        pairs = sorted({p.fitness for p in front})
        spacing = dict.fromkeys(pairs, 0.0)
        for axis in (0, 1):
            values = sorted(pairs, key=lambda pair: pair[axis])
            span = values[-1][axis] - values[0][axis]
            if span > 0:
                spacing[values[0]] = spacing[values[-1]] = float("inf")
                for i in range(1, len(values) - 1):
                    spacing[values[i]] += (values[i+1][axis] - values[i-1][axis]) / span
        for p in front:
            p.pareto_rank = rank
            p.crowding = spacing[p.fitness]
        rng.shuffle(front)
        front.sort(key=lambda p: -p.crowding)
        if rank == 0 and goal is not None:
            # Choose the block's representative from its Pareto front.
            axis = GOALS.index(goal)
            leader = min(front, key=lambda p: p.fitness[axis])
            front.remove(leader)
            front.insert(0, leader)
        ordered.extend(front)
        selected_ids = {id(p) for p in front}
        remaining = [p for p in remaining if id(p) not in selected_ids]
        rank += 1
    return ordered


def pareto_tournament(ordered, count, tournament_size, winners):
    # Preserve one representative of each non-dominated tradeoff first.
    survivors = []
    seen = set()
    for p in ordered:
        if p.pareto_rank == 0 and p.fitness not in seen:
            survivors.append(p)
            seen.add(p.fitness)
            if len(survivors) == count:
                return survivors
    while len(survivors) < count:
        k = min(tournament_size, len(ordered))
        tournament = rng.choice(ordered, size=k, replace=False)
        tournament = sorted(tournament, key=lambda p: (p.pareto_rank, -p.crowding))
        survivors.extend(tournament[:min(winners, count - len(survivors))])
    return survivors


def save_policies(path, population):
    # Front members may be repeated by tournaments. Save each individual once.
    front = list(dict.fromkeys(p for p in population if p.pareto_rank == 0))
    with NamedTemporaryFile(dir=path.parent, suffix=".npz", delete=False) as tmp:
        stage = Path(tmp.name)
    try:
        np.savez_compressed(stage, version=1, goals=[p.goal_weights for p in front],
                            controllers=[p.controller_weights for p in front],
                            objectives=[p.fitness for p in front])
        stage.replace(path)
    finally:
        stage.unlink(missing_ok=True)


class Individual_Game:

    def __init__(self, game, goal, weights):
        self.game = copy.deepcopy(game)
        self.state = self.game.get_state()
        self.fitness = 0
        self.goal = goal
        self.genome = weights.copy()  # Inherited rule for selecting dig or wells.
        self.pick_move()

    def pick_move(self):
        features = policy_inputs(self.state)
        features.extend(float(self.state["current"] == piece) for piece in PIECES)
        weights = self.genome[GOALS.index(self.goal)]
        self.strategy = "wells" if np.dot(weights, features) >= 0 else "dig"
        self.move = choose_moves(self.game, self.strategy)

    def eval_fitness(self):
        if self.move is None:
            self.fitness = (float("inf"), float("inf"))
            return

        before = self.game.pieces_placed
        rotation, column = self.move
        self.game.place(rotation, column)
        self.state = self.game.get_state()

        if self.game.game_over or self.game.pieces_placed != before + 1:
            self.fitness = (float("inf"), float("inf"))
            return

        # Keep your fitness calculation below this.
        heights = self.state["column_heights"]
        self.fitness = (float(np.var(heights)), float(np.mean(heights)))   # low = flat board

class Individual_Strategizer:

    def __init__(self, game):
        # negative/positive direction = number of inputs left/right 
        # negative/positive rotation = number of rotations clockwise/counterclockwise
        self.game = copy.deepcopy(game) # FIX: own copy, so building strategizers doesn't drop pieces on the shared start board
        self.goal_weights = rng.normal(size=5)
        self.controller_weights = rng.normal(size=(len(GOALS), 5 + len(PIECES)))
        self.record_initial()

        # Original heuristic, now decided by the inherited goal weights.
        # flat board, prioritize reducing height (bounds picked on whim)
        self.goal = None  # Set from the current board when the block starts.
        self.fitness = 0

    def record_initial(self):
        self.initial_state = self.game.get_state()
        self.initial_heights = self.initial_state["column_heights"]
        self.initial_var = np.var(self.initial_heights)
        self.initial_avg = np.mean(self.initial_heights)

    def play(self):
        # FIX: the controller session used to run only once, in __init__, so later generations
        # (and strategy flips) never played anything. Now it runs every generation, from this
        # strategizer's current board, with its current strategy.
        self.record_initial()
        self.pick_strategy()  # Decide once, at the beginning of every block.
        self.game = evolutionary_controller(self.game, self.goal, self.controller_weights)[1] # genome becomes best board at end of controller session

    def eval_fitness(self):
        post_state = self.game.get_state()
        post_heights = post_state["column_heights"]
        post_var = np.var(post_heights)
        post_avg = np.mean(post_heights)


        # Earlier fitness notes (the two objectives now use Pareto dominance).
        # TODO: introduce loss function with true weighting opposed to if/else calculating fitness
            # then change weights based on strategy
        # inverted fitness - lower fitness = more fit
        # lower final varianced compared to inital is more fit 
        completed = self.game.pieces_placed - self.initial_state["pieces_placed"]
        if self.game.game_over or completed != BLOCK_PIECES:
            self.fitness = (float("inf"), float("inf"))
        else:
            self.fitness = (float(post_var), float(post_avg))

    # TODO: hook in tactician to pick_strategy function
    def pick_strategy(self):
        # noisy board, prioritize reducing variance
        signal = np.dot(self.goal_weights, policy_inputs(self.initial_state))
        self.goal = "flatness" if signal >= 0 else "lowness"


def evolutionary_controller(game=None, goal="flatness", controller_weights=None):
    # The winning controller writes its policy back to the calling strategizer.
    if goal not in GOALS:
        raise ValueError(f"Unknown block goal: {goal}")
    if game is None:
        game = NESTetris(seed=SEED)
    if controller_weights is None:
        controller_weights = rng.normal(size=(len(GOALS), 5 + len(PIECES)))
    axis = GOALS.index(goal)

    # Evolutinary Algorithm framework copied from assignment 6
    total_generations=BLOCK_PIECES - 1; num_parents=10; num_children=10;  # Initialization places piece one.
    tournament_size=4; num_tournament_winners=2;

    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = 99999 # fitness of best genome so far
    fitness_over_time = np.zeros((total_generations, 2))
    solutions_over_time = []
    diversity_over_time = np.zeros(total_generations)

    # the initialization proceedure
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        weights = controller_weights.copy()
        if i:
            weights[axis] += rng.normal(0, 0.20, size=weights.shape[1])
        population.append(Individual_Game(game, goal, weights)) # generate new random individuals as parents

    # get population fitness
    for i in range(len(population)):
        population[i].eval_fitness() # evaluate the fitness of each parent
    population = pareto_order(population, goal)
    if not population:
        return solutions_over_time, game

    # total_generations in this case refers to as pieces to iterate through
    for generation_num in range(total_generations): # repeat

        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:

            # inheretance
            [parent1, parent2] = rng.choice(population, size=2) # pick 2 random parents
            child1 = copy.deepcopy(parent1) # initialize children as perfect copies of their parents
            child2 = copy.deepcopy(parent2)

            # crossover
            # N/A  

            # mutation
            for this_child in [child1,child2]: 
                this_child.genome[axis] += rng.normal(0, 0.20, size=this_child.genome.shape[1])
                this_child.pick_move()

            new_children.extend((child1,child2)[:num_children-len(new_children)]) # add children to the new_children list

        # the assessement procedure
        for i in range(len(new_children)):
            new_children[i].eval_fitness() # assign fitness to each child and continue game

        # selection procedure
        population = new_children   # only children survive: every survivor dropped this generation's piece
        # Earlier scalar selection notes:
        # sort the full population by each individual's fitness (from highers to lowest)
        # reverse equals false, as lowest fitness values should win
        population = pareto_order(population, goal)
        if not population:
            return solutions_over_time, game

        # tournament selection
        population = pareto_tournament(population, num_parents, tournament_size, num_tournament_winners)

        # record keeping

        solution = population[0].game.render_text()                 # update best solution records
        solution_fitness = population[0].fitness
        fitness_over_time[generation_num] = solution_fitness # record the fitness of the current best over evolutionary time
        solutions_over_time.append(population[0].game.render_text())


        # removed diversity as this implementation doesnt apply, also not needed for this version


    controller_weights[:] = population[0].genome
    return solutions_over_time, population[0].game


def evolutionary_strategizer(total_generations=10, num_parents=20, num_children=20, tournament_size=8, num_tournament_winners=4, policy_path=POLICY_PATH):

    print("what it do")
    if min(num_parents, num_children, tournament_size, num_tournament_winners) < 1:
        raise ValueError("Population and tournament sizes must be positive")
    if total_generations < 0:
        raise ValueError("total_generations must be nonnegative")
    # Evolutinary Algorithm framework copied from assignment 6

    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = 99999 # fitness of best genome so far
    fitness_over_time = np.zeros((total_generations, 2))
    solutions_over_time = []
    diversity_over_time = np.zeros(total_generations)
    game = NESTetris(seed=SEED)
    print("what it do4")
    # the initialization proceedure
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Individual_Strategizer(game)) # generate new random individuals as parents

    if policy_path is not None:
        policy_path = Path(policy_path)
        if policy_path.is_file():
            with np.load(policy_path, allow_pickle=False) as saved:
                goals, controllers = saved["goals"], saved["controllers"]
                if (int(saved["version"]) != 1 or goals.ndim != 2 or goals.shape[1] != 5
                        or not len(goals) or controllers.shape != (len(goals), 2, 5 + len(PIECES))
                        or not np.isfinite(goals).all() or not np.isfinite(controllers).all()):
                    raise ValueError(f"Incompatible policy checkpoint: {policy_path}")
                for i, individual in enumerate(population):
                    individual.goal_weights = goals[i % len(goals)].copy()
                    individual.controller_weights = controllers[i % len(goals)].copy()
            print(f"Loaded {len(goals)} Pareto policies from {policy_path.name}")

    # get population fitness
    for i in range(len(population)):
        population[i].play()
        population[i].eval_fitness() # evaluate the fitness of each parent
    population = pareto_order(population)
    if not population:
        print("No branch completed the initial 10-piece block.")
        return solutions_over_time, game
    if policy_path is not None:
        save_policies(policy_path, population)
    game = population[0].game
    print("what it do3")
    # total_generations in this case refers to as pieces to iterate through
    for generation_num in range(total_generations): # repeat

        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:

            # inheretance
            [parent1, parent2] = rng.choice(population, size=2) # pick 2 random parents
            child1 = copy.deepcopy(parent1) # initialize children as perfect copies of their parents
            child2 = copy.deepcopy(parent2)

            # crossover
            # N/A

            # mutation - likely will depend on tactician input 
                # randomly select few children to flip strategy
                # TODO: punish in tactician if alternative strategy out performs chosen strategy
            for this_child in [child1,child2]:
                if rng.random() <= 0.25:
                    # Mutate the inherited decision rule; play() chooses the block goal.
                    this_child.goal_weights += rng.normal(0, 0.25, size=this_child.goal_weights.shape)

            new_children.extend((child1,child2)[:num_children-len(new_children)]) # add children to the new_children list

        # the assessement procedure
        for i in range(len(new_children)):
            new_children[i].play()
            new_children[i].eval_fitness() # assign fitness to each child and continue game

        # selection procedure
        population = new_children   # only children survive: every survivor dropped this generation's piece
        # Earlier scalar selection notes:
        # sort the full population by each individual's fitness (from highers to lowest)
        # reverse equals false, as lowest fitness values should win
        population = pareto_order(population)
        if not population:
            print(f"No branch completed block {generation_num + 2}.")
            return solutions_over_time, game

        # tournament selection
        population = pareto_tournament(population, num_parents, tournament_size, num_tournament_winners)

        # record keeping

        if policy_path is not None:
            save_policies(policy_path, population)
        game = population[0].game                             # update starting board
        solution_fitness = population[0].fitness
        fitness_over_time[generation_num] = solution_fitness # record the fitness of the current best over evolutionary time
        solutions_over_time.append(game.render_text())

        for i, individual in enumerate(population):
            print(f"Generation {generation_num + 1}, Game {i + 1}: "
                f"{individual.game.pieces_placed} pieces placed | goal={individual.goal} | "
                f"variance={individual.fitness[0]:.2f} mean_height={individual.fitness[1]:.2f} | "
                f"Pareto front {individual.pareto_rank + 1}")

        print("what it do2")
        # removed diversity as this implementation doesnt apply, also not needed for this version


    return solutions_over_time, population[0].game

if __name__ == "__main__":
    print("starting")
    sot, game = evolutionary_strategizer()

    for sol in sot:
        print(sol)
        print()
