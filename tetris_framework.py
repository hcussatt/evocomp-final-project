"""Evolve reusable neural controllers through real games, without retries."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from itertools import permutations
import random
from nes_tetris import NESTetris, ORIENTATIONS

ACTIONS = ('left', 'right', 'cw', 'ccw', 'noop')
BUTTONS = {'left': 'left', 'right': 'right', 'cw': 'A', 'ccw': 'B', 'noop': None, 'down': 'down'}


def cells(pose):
    oid, x, y = pose
    return frozenset((y + dy, x + dx) for dy, dx in ORIENTATIONS[oid][1])


def error(actual, target):
    """Minimum matched-cell Manhattan distance; zero means exact placement."""
    a, b = sorted(cells(actual)), sorted(cells(target))
    return min(sum(abs(r-s) + abs(c-d) for (r,c),(s,d) in zip(a,p))
               for p in permutations(b))


def execute(game, actions, on_frame=None, max_control_frames=6000):
    """Poll the policy throughout descent; score only after the piece locks."""
    landing = None
    start_frame = game.frames
    def advance(button=None, controlling=False):
        nonlocal landing
        if controlling and game.frames-start_frame >= max_control_frames:
            raise RuntimeError('Piece exceeded control frame limit; no forced drop or retry performed')
        if game._active():
            landing = (game.orientation_id, *game.piece_xy)
        game.frame(*(() if button is None else (button,)))
        if on_frame:
            on_frame(game)
    for action in actions:
        if not game._active() or game.game_over:
            break
        advance(BUTTONS[action], controlling=True)
        if game._active() and not game.game_over:
            # Rotation/horizontal actions are taps. Consecutive down decisions
            # hold down continuously, but NEVER give up control of the piece.
            advance('down' if action == 'down' else None, controlling=True)
        if not game._active() or game.game_over:
            break
    if game._active() and not game.game_over:
        raise RuntimeError('Controller stopped before lock; no automatic drop performed')
    if landing is None:
        raise RuntimeError('No active piece to control')
    for _ in range(game.MAX_WAIT_FRAMES):
        if game._active() or game.game_over:
            break
        advance()
    else:
        raise RuntimeError('Next piece did not spawn')
    game.pieces_placed += 1
    return landing


# Inputs: heights (10), sorted cell displacement (8), piece identity (7),
# current/target orientation (19 each), remaining watchdog frame fraction (1).
INPUTS, HIDDEN, OUTPUTS = 64, 12, 6
POLICY_ACTIONS = ACTIONS + ('down',)
NETWORK = [INPUTS,HIDDEN,OUTPUTS]
PARAMETERS = sum(a*b+b for a,b in zip(NETWORK,NETWORK[1:]))

def configure_network(architecture):
    global NETWORK, PARAMETERS
    if len(architecture)<2 or architecture[0]!=INPUTS or architecture[-1]!=OUTPUTS or any(type(x) is not int or x<1 for x in architecture):
        raise ValueError("Architecture must start with 64, end with 6, and contain positive integer widths")
    NETWORK = list(architecture)
    PARAMETERS = sum(a*b+b for a,b in zip(NETWORK,NETWORK[1:]))
KINDS = ('T','J','Z','O','S','L','I')


def features(game, target, remaining):
    current = sorted(cells((game.orientation_id, *game.piece_xy)))
    desired = sorted(cells(target))
    displacement = [(d-c)/20 for a,b in zip(current,desired) for c,d in zip(a,b)]
    return np.array([* (game.column_heights()/20), *displacement,
        *[float(game.current_piece == k) for k in KINDS],
        *[float(game.orientation_id == i) for i in range(19)],
        *[float(target[0] == i) for i in range(19)], remaining], dtype=float)


class Controller:
    def __init__(self, weights):
        self.architecture = tuple(NETWORK)
        self.weights = np.asarray(weights, dtype=float).copy()
        if self.weights.shape != (PARAMETERS,) or not np.isfinite(self.weights).all():
            raise ValueError('Invalid controller weights')
    def action(self, observation):
        offset = 0
        value = observation
        for layer,(nin,nout) in enumerate(zip(self.architecture,self.architecture[1:])):
            matrix = self.weights[offset:offset+nin*nout].reshape(nin,nout)
            offset += nin*nout
            bias = self.weights[offset:offset+nout]
            offset += nout
            value = value @ matrix + bias
            if layer < len(self.architecture)-2:
                value = np.tanh(value)
        return POLICY_ACTIONS[int(np.argmax(value))]
    def inputs(self, game, target, frame_limit, history):
        start_frame = game.frames
        while game._active() and not game.game_over:
            remaining = max(0.0, 1.0-(game.frames-start_frame)/frame_limit)
            action = self.action(features(game, target, remaining))
            history.append(action)
            yield action


def choose_target(game, rng):
    # Geometry only: no emulator trials, clones, restore, or solution sequence.
    poses = [game._resolve(rot,x) for rot,x in game.legal_placements()]
    poses = [p for p in poses if all(r >= 0 for r,c in cells(p))]
    return rng.choice(poses) if poses else None


class OnlineEvolution:
    """Select from recent scored policies and create one mutant per placement."""
    def __init__(self, weights, rng, capacity, sigma):
        self.weights = Controller(weights).weights
        self.rng, self.capacity, self.sigma = rng, capacity, sigma
        self.archive = []
        self.controller_id = 1
        self.parent_id = 0
        self.mutations = 0

    def observe(self, distance):
        self.archive.append((self.controller_id, float(distance), self.weights.copy()))
        self.archive = self.archive[-self.capacity:]
        indices = self.rng.choice(len(self.archive), size=min(3,len(self.archive)), replace=False)
        parent = min((self.archive[i] for i in indices), key=lambda x:x[1])
        child = parent[2].copy()
        mask = self.rng.random(PARAMETERS) < .2
        if not mask.any():
            mask[self.rng.integers(PARAMETERS)] = True
        child[mask] += self.rng.normal(0,self.sigma,int(mask.sum()))
        if np.array_equal(child,parent[2]):
            k = int(np.flatnonzero(mask)[0])
            child[k] = np.nextafter(child[k],np.inf)
        changed = int(np.count_nonzero(child != parent[2]))
        self.parent_id = parent[0]
        self.controller_id += 1
        self.weights = child
        self.mutations += 1
        return changed


def save_policy(path, weights, **metadata):
    payload = dict(format_version=3,actions=list(POLICY_ACTIONS),
                   architecture=list(NETWORK),weights=weights.tolist(),**metadata)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(payload))
    temp.replace(path)


def run(args, viewer=None):
    hidden = getattr(args,"hidden","12")
    configure_network([INPUTS]+([] if hidden == "linear" else [int(x) for x in hidden.split(",")])+[OUTPUTS])
    rng = np.random.default_rng(args.seed)
    weights = rng.normal(0,.3,PARAMETERS)
    if args.load:
        checkpoint = json.loads(Path(args.load).read_text())
        if checkpoint.get('format_version',1) < 2:
            print('Loading older weights as a warm start: drop now means temporary down; timing feature has changed.', flush=True)
        configure_network(checkpoint.get('architecture',[64,12,6]))
        weights = Controller(checkpoint['weights']).weights

    evolution = OnlineEvolution(weights,rng,args.population,args.sigma)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    evaluate = getattr(args,'evaluate',False)
    if not evaluate:
        save_policy(out/'initial_controller.json',weights,score_scope='initial_untrained')
    total_attempts = 0
    best_score = float('inf')
    summaries = []
    with (out/'attempts.csv').open('w',newline='') as af, (out/'games.csv').open('w',newline='') as gf:
        attempts = csv.DictWriter(af, fieldnames=['attempt','game','generation','controller','parent_controller','piece','target','actual','error','fitness','next_controller','next_parent','mutated_weights','inputs'])
        games = csv.DictWriter(gf, fieldnames=['game','first_mutation','last_mutation','pieces','mean_error','exact','fitness','end'])
        attempts.writeheader(); games.writeheader()
        for game_index in range(args.games):
            first_mutation = evolution.mutations
            # Every game is new; within a game the emulator only moves forward.
            game = NESTetris(seed=args.seed+game_index)
            target_rng = random.Random(args.seed+100000+game_index)
            errors = []
            reason = 'piece_limit'
            try:
                for turn in range(1,args.pieces+1):
                    if game.game_over:
                        reason = 'game_over'; break
                    target = choose_target(game,target_rng)
                    if target is None:
                        reason = 'no_legal_target'; break
                    controller = Controller(evolution.weights)
                    controller_id, parent_id = evolution.controller_id, evolution.parent_id
                    generation = total_attempts+1
                    if viewer:
                        viewer.target = target
                        viewer.status = f'Game {game_index+1}/{args.games} | Mutation {evolution.mutations} | Piece {turn}'
                        viewer.draw(game)
                    history = []
                    actual = execute(game,controller.inputs(game,target,args.max_piece_frames,history),
                                     viewer.frame if viewer else None, args.max_piece_frames)
                    distance = error(actual,target)/4.0
                    errors.append(distance)
                    total_attempts += 1
                    if not evaluate and distance < best_score:
                        best_score = distance
                        save_policy(out/'best_controller.json',controller.weights,
                            game=game_index+1,attempt=generation,piece_error=best_score,
                            controller=controller_id,score_scope='single_piece')
                    # Exactly one update AFTER scoring, even on the final piece
                    # of a game. The new weights carry into the next game.
                    changed = 0 if evaluate else evolution.observe(distance)
                    if not evaluate:
                        save_policy(out/'current_controller.json',evolution.weights,
                            completed_attempts=evolution.mutations,controller=evolution.controller_id,
                            parent_controller=evolution.parent_id,score_scope='not_yet_evaluated')
                    attempts.writerow(dict(attempt=generation,game=game_index+1,generation=generation,
                        controller=controller_id,parent_controller=parent_id,piece=turn,
                        target=target,actual=actual,error=distance,fitness=-distance,
                        next_controller=evolution.controller_id,next_parent=evolution.parent_id,
                        mutated_weights=changed,inputs=' '.join(history)))
                    af.flush()
                    if viewer:
                        viewer.last_error = distance
                        viewer.target = None
                        viewer.draw(game)
                    if game.game_over:
                        reason = 'game_over'; break
            finally:
                game.close()
            # A controller with no attempts must not win. Flatness/survival are
            # not objectives. Error is mean matched-cell distance per attempt.
            mean_error = float(np.mean(errors)) if errors else 100.0
            row = dict(game=game_index+1,first_mutation=first_mutation,last_mutation=evolution.mutations,
                       pieces=len(errors),mean_error=mean_error,exact=sum(e==0 for e in errors),
                       fitness=-mean_error,end=reason)
            summaries.append(row); games.writerow(row); gf.flush()
            print(row,flush=True)
    return summaries


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--render',action='store_true')
    p.add_argument('--pace',action='store_true',help='Compatibility flag; graphics are paced by default')
    p.add_argument('--games',type=int,default=100)
    p.add_argument('--hidden',default='12',help='Hidden widths: linear, 12, or 32,16; loaded checkpoint overrides this')
    p.add_argument('--evaluate',action='store_true',help='Play a frozen policy without mutation')
    p.add_argument('--pieces',type=int,default=200,help='Safety limit per game')
    p.add_argument('--population',type=int,default=10,help='Number of recent scored controllers retained for parent selection')
    p.add_argument('--seed',type=int,default=67)
    p.add_argument('--max-piece-frames',type=int,default=6000,help='Watchdog only; abort on timeout without forcing a drop')
    p.add_argument('--fps',type=int,default=60)
    p.add_argument('--sigma',type=float,default=.15)
    p.add_argument('--load',help='Warm start from current_controller.json or best_controller.json')
    p.add_argument('--output',default=str(Path(__file__).parent/'controller_results'))
    args=p.parse_args(argv)
    if min(args.games,args.pieces,args.max_piece_frames,args.fps)<1 or args.population<2 or not np.isfinite(args.sigma) or args.sigma<=0:
        p.error('Counts must be positive, population >= 2, and sigma positive and finite')
    viewer=None
    try:
        if args.render:
            from tetris_framework_graphical import BoardViewer
            viewer=BoardViewer(args.fps)
        results=run(args,viewer)
        print(f'Completed {len(results)} games. Results: {args.output}')
        if viewer:
            viewer.status=f'Finished {len(results)} games — close window to exit'
            viewer.pump(force=True)
            viewer.root.mainloop()
    except KeyboardInterrupt:
        print('Stopped. Completed attempts/games and controller weights are saved.')
        raise SystemExit(130)
    finally:
        if viewer:
            viewer.close()

if __name__=='__main__':
    main()
