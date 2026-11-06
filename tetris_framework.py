"""
tetris_framework.py: basic loop for playing NES Tetris one piece at a time.

Put this file in the same folder as nes_tetris.py and run it.
Edit the SETTINGS section and the choose_moves() function to experiment.
"""

import random
import copy
import numpy as np
from nes_tetris import NESTetris

SEED = 67

# ----------------------------------------------------------------------------
# MUTATION LOGIC - Currently set to be totally random, but has capability to enter target_column decisions 
# ----------------------------------------------------------------------------
def choose_moves(state):
    """
    Look at the game state and return a list of moves for the current piece.

    state["column_heights"]  list of 10 ints, left to right (0 = empty column)
    state["current"]         current piece: "T", "J", "Z", "O", "S", "L" or "I"
    state["next"]            the next piece
    state["lines"], state["score"], state["level"]

    Valid moves: "left", "right", "cw" (rotate clockwise), "ccw" (counter-clockwise)
    The piece drops automatically after your moves.

    This example: random rotation, then shift toward the lowest column.
    """
    heights = state["column_heights"]
    # target_column = heights.index(min(heights))       # lowest column (0-9)
    target_column = random.randint(0,9)

    moves = []
    moves += ["cw"] * random.randint(0, 3)             # 0-3 clockwise rotations

    # pieces spawn around column 5, so shift left or right from there
    shift = target_column - 5
    if shift < 0:
        moves += ["left"] * (-shift)
    else:
        moves += ["right"] * shift
    return moves


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
class Individual:
    
    def __init__(self):
        self.game = NESTetris(seed=SEED)
        self.state = self.game.get_state()
        self.genome = choose_moves(self.state)
        self.fitness = 0
        
    def eval_fitness(self):
        self.game.play_moves(self.genome)
        self.state = self.game.get_state()
        self.fitness = np.var(self.state["column_heights"])   # low = flat board


def evolutionary_algorithm(total_generations=20, num_parents=10, num_children=10, genome_length=10, tournament_size=4, num_tournament_winners=2):
    
    # Evolutinary Algorithm framework copied from assignment 6
    
    # initialize record keeping
    solution = None # best genome so far
    solution_fitness = 99999 # fitness of best genome so far
    fitness_over_time = np.zeros(total_generations)
    solutions_over_time = []
    diversity_over_time = np.zeros(total_generations)
    
    # the initialization proceedure
    population = [] # keep population of individuals in a list
    for i in range(num_parents): # only create parents for initialization (the mu in mu+lambda)
        population.append(Individual()) # generate new random individuals as parents
    
    # get population fitness
    for i in range(len(population)):
        population[i].eval_fitness() # evaluate the fitness of each parent

    # total_generations in this case refers to as pieces to iterate through
    for generation_num in range(total_generations): # repeat
        
        # the modification procedure
        new_children = [] # keep children separate for now (lambda in mu+lambda)
        while len(new_children) < num_children:
            
            # inheretance
            [parent1, parent2] = np.random.choice(population, size=2) # pick 2 random parents
            child1 = copy.deepcopy(parent1) # initialize children as perfect copies of their parents
            child2 = copy.deepcopy(parent2)
            
            # crossover
            # N/A

            # mutation
            for this_child in [child1,child2]:
                this_child.genome = choose_moves(this_child.state)
        
            new_children.extend((child1,child2)) # add children to the new_children list
            
        # the assessement procedure
        for i in range(len(new_children)):
            new_children[i].eval_fitness() # assign fitness to each child and continue game

        # selection procedure
        population = new_children   # only children survive: every survivor dropped this generation's piece
        population = sorted(population, key=lambda individual: individual.fitness) # sort the full population by each individual's fitness (from highers to lowest)
        
        # tournament selection
        new_population = []
        new_population.append(population[0])
        while len(new_population) < num_parents:
            tournament = np.random.choice(population, size=tournament_size, replace=False)
            tournament = sorted(tournament, key=lambda individual: individual.fitness) # reverse equals false, as lowest fitness values should win
            new_population.extend(tournament[:num_tournament_winners])    
        population = new_population
        
        # record keeping
        
        solution = population[0].game.render_text()                 # update best solution records
        solution_fitness = population[0].fitness
        fitness_over_time[generation_num] = solution_fitness # record the fitness of the current best over evolutionary time
        solutions_over_time.append(population[0].game.render_text())
        
        
        # removed diversity as this implementation doesnt apply, also not needed for this version
                
        
    return fitness_over_time, solutions_over_time, diversity_over_time

fot, sot, dot = evolutionary_algorithm()

for sol in sot:
    print(sol)
    print()

