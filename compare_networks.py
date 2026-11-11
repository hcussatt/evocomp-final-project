"""Train three architectures piece by piece and evaluate initial/final policies."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from pathlib import Path
import subprocess
import sys

MODELS = [('linear','linear'),('small','12'),('deep','32,16')]

def launch(out, hidden, games, seed, load=None):
    out.mkdir(parents=True,exist_ok=True)
    command=[sys.executable,str(Path(__file__).with_name('tetris_framework.py')),
             '--hidden',hidden,'--games',str(games),'--seed',str(seed),'--output',str(out)]
    if load:
        command += ['--load',str(load),'--evaluate']
    with (out/'console.txt').open('w') as log:
        subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True)
    with (out/'games.csv').open() as f:
        if sum(1 for _ in csv.DictReader(f)) != games:
            raise RuntimeError(f'Incomplete run in {out}; stopping comparison')

def summarize(root):
    rows=[]
    for model,_ in MODELS:
        for phase in ['train','initial_eval','final_eval']:
            folder=root/model/phase
            data=list(csv.DictReader((folder/'attempts.csv').open()))
            errors=[float(x['error']) for x in data]
            if not errors:
                raise ValueError(f'No completed attempts in {folder}')
            games=list(csv.DictReader((folder/'games.csv').open()))
            rows.append(dict(model=model,phase=phase,games=len(games),attempts=len(errors),
                mean_error=sum(errors)/len(errors),exact_rate=sum(x==0 for x in errors)/len(errors),
                first100_error=sum(errors[:100])/len(errors[:100]),
                last100_error=sum(errors[-100:])/len(errors[-100:])))
    with (root/'summary.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    for row in rows: print(row)
    return rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--games',type=int,default=100)
    p.add_argument('--jobs',type=int,default=1,help='Concurrent model experiments; use 1 to limit CPU load')
    p.add_argument('--eval-games',type=int,default=5)
    p.add_argument('--seed',type=int,default=67)
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'nn_comparison_new')
    p.add_argument('--summarize-only',action='store_true')
    args=p.parse_args()
    if min(args.games,args.eval_games,args.jobs)<1:p.error('Game counts must be positive')
    if not args.summarize_only:
        def experiment(model_hidden):
            model,hidden=model_hidden
            print('Training',model,flush=True)
            base=args.output/model
            launch(base/'train',hidden,args.games,args.seed)
            for phase,checkpoint in [('initial_eval','initial_controller.json'),('final_eval','current_controller.json')]:
                print('Evaluating',model,phase,flush=True)
                launch(base/phase,hidden,args.eval_games,args.seed+10000,base/'train'/checkpoint)
        with ThreadPoolExecutor(max_workers=min(args.jobs,len(MODELS))) as pool:
            list(pool.map(experiment,MODELS))
    summarize(args.output)

if __name__=='__main__':main()
