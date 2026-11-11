"""Regression checks using the real emulator; no snapshot/restore allowed."""
import argparse
import csv
import json
import tempfile
from pathlib import Path
import numpy as np
import tetris_framework as f


def late_rotation_test():
    game = f.NESTetris(seed=67)
    events=[]
    def controls():
        # Descend well below spawn BEFORE rotating, then move and resume down.
        while game._active() and game.piece_xy[1] < 12:
            yield 'down'
        before = game.orientation_id
        row = game.piece_xy[1]
        yield 'cw'
        events.append((row,before,game.orientation_id))
        yield 'left'
        while game._active():
            yield 'down'
    try:
        pose=f.execute(game,controls())
        assert events and events[0][0]>=12
        assert events[0][1]!=events[0][2], events
        assert game.pieces_placed==1
        assert pose[0]==events[0][2]
        print('PASS: down to row 12, rotate, move left, resume down, lock once.', events)
    finally:
        game.close()
    # Never silently force a drop when control ends or the watchdog fires.
    game=f.NESTetris(seed=67)
    try:
        try:
            f.execute(game,iter(['noop']))
        except RuntimeError as e:
            assert 'no automatic drop' in str(e)
        else:
            raise AssertionError('Unexpected forced completion')
        assert game.pieces_placed==0 and game._active()
        import itertools
        try:
            f.execute(game,itertools.repeat('noop'),max_control_frames=2)
        except RuntimeError as e:
            assert 'no forced drop' in str(e)
        else:
            raise AssertionError('Watchdog failed')
        assert game.pieces_placed==0
    finally:
        game.close()


def test():
    # Generic evaluator must preserve the original small-network computation.
    rng=np.random.default_rng(44)
    x=rng.normal(size=f.INPUTS)
    for shape in [[64,6],[64,12,6],[64,32,16,6]]:
        f.configure_network(shape)
        w=rng.normal(0,.3,f.PARAMETERS)
        net=f.Controller(w)
        assert net.action(x) in f.POLICY_ACTIONS
        if shape==[64,12,6]:
            h=np.tanh(x @ w[:768].reshape(64,12)+w[768:780])
            expected=f.POLICY_ACTIONS[int(np.argmax(h @ w[780:852].reshape(12,6)+w[852:]))]
            assert net.action(x)==expected
    f.configure_network([64,12,6])
    late_rotation_test()
    # Known scoring examples: exact, translation, wrong orientation.
    assert f.error((2,5,18),(2,5,18)) == 0
    assert f.error((2,5,18),(2,6,18))/4 == 1
    assert f.error((2,5,18),(3,5,18)) > 0
    rng=np.random.default_rng(7)
    evo=f.OnlineEvolution(rng.normal(0,.3,f.PARAMETERS),rng,2,.15)
    old=evo.weights.copy()
    assert evo.observe(5)>0 and not np.array_equal(old,evo.weights)
    assert evo.parent_id==1 and evo.mutations==1
    assert evo.observe(1)>0 and evo.parent_id==2
    # Recent archive ages out an old lucky result, instead of keeping it forever.
    evo.observe(10)
    assert [x[0] for x in evo.archive]==[2,3]
    # Guard against regression to repeated trials or restoring the live game.
    original=f.NESTetris
    class ForwardOnly(original):
        instances=[]
        def __init__(self,*args,**kwargs):
            self.running=False
            super().__init__(*args,**kwargs)
            self.running=True
            self.instances.append(self)
        def reset(self,*args,**kwargs):
            assert not self.running, 'Reset inside a live game'
            return super().reset(*args,**kwargs)
        def snapshot(self):
            raise AssertionError('No snapshots allowed')
        def restore(self,*args):
            raise AssertionError('No restoring allowed')
        def clone(self):
            raise AssertionError('No trial clones allowed')
        def frame(self,*args):
            previous=self.frames
            super().frame(*args)
            assert self.frames == previous+1
    f.NESTetris=ForwardOnly
    try:
        with tempfile.TemporaryDirectory() as d:
            args=argparse.Namespace(seed=67,population=2,load=None,output=d,games=4,
                                    pieces=2,max_piece_frames=6000,sigma=.15)
            rows=f.run(args)
            assert len(rows)==len(ForwardOnly.instances)==4
            assert [r['first_mutation'] for r in rows]==[0,2,4,6]
            assert [r['last_mutation'] for r in rows]==[2,4,6,8]
            assert all(r['pieces']==2 for r in rows)
            assert all(g.pieces_placed==2 for g in ForwardOnly.instances)
            attempts=list(csv.DictReader(open(Path(d)/'attempts.csv')))
            assert len(attempts)==8
            assert [int(r['attempt']) for r in attempts]==list(range(1,9))
            assert all(int(r['next_controller'])==int(r['controller'])+1 for r in attempts)
            assert all(int(r['mutated_weights'])>0 for r in attempts)
            checkpoint_current=json.loads((Path(d)/'current_controller.json').read_text())
            assert checkpoint_current['completed_attempts']==8
            assert checkpoint_current['controller']==9
            assert any(float(r['error'])>0 for r in attempts)
            checkpoint=json.loads((Path(d)/'best_controller.json').read_text())
            saved=f.Controller(checkpoint['weights'])
            assert saved.action(np.zeros(f.INPUTS)) in f.POLICY_ACTIONS
            args.load=str(Path(d)/'best_controller.json')
            args.output=str(Path(d)/'continued')
            args.games=1
            assert len(f.run(args))==1
            args.evaluate=True
            args.output=str(Path(d)/'frozen')
            frozen=f.run(args)
            assert frozen[0]['first_mutation']==frozen[0]['last_mutation']==0
            frozen_rows=list(csv.DictReader(open(Path(args.output)/'attempts.csv')))
            assert [int(r['attempt']) for r in frozen_rows]==[1,2]
            assert all(int(r['mutated_weights'])==0 for r in frozen_rows)
    finally:
        f.NESTetris=original
    print('PASS: scoring, mutation after every piece, cross-game continuity, single attempts, no rewinds, four games, checkpoint reload.')

if __name__=='__main__':
    test()
