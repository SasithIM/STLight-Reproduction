"""Run the partner's long jobs sequentially, stopping on errors.

Use --wait-for-smoke-pid to continue after an already running smoke process.
Status and complete console transcripts are in work_dirs/partner_pipeline.
"""
import argparse
import ctypes
import datetime
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'work_dirs' / 'partner_pipeline'


def status(message):
    text = f'{datetime.datetime.now().isoformat(timespec="seconds")} {message}'
    (OUT / 'status.txt').write_text(text + '\n', encoding='utf-8')
    with (OUT / 'history.txt').open('a', encoding='utf-8') as f:
        f.write(text + '\n')
    print(text, flush=True)


def run(name, args):
    status(f'Running {name}')
    with (OUT / f'{name}.txt').open('w', encoding='utf-8') as log:
        child = subprocess.Popen([sys.executable, '-u', *args], cwd=ROOT,
                                 stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 text=True, encoding='utf-8', errors='replace')
        for line in child.stdout:
            log.write(line)
            log.flush()
            if re.search(r'(?:train_loss\w*|val_loss)\s*[=:]\s*(?:nan|inf)\b', line, re.I):
                child.terminate()
                child.wait()
                raise RuntimeError(f'Non-finite loss in {name}; see its transcript')
        if child.wait() != 0:
            raise RuntimeError(f'{name} failed with exit code {child.returncode}')


def train(name, config):
    if (ROOT / 'work_dirs' / name / 'checkpoints' / 'last.ckpt').exists():
        raise RuntimeError(f'{name} already has a checkpoint; refusing to overwrite it')
    run(name, ['tools/train.py', '-d', 'mmnist', '-c', config,
               '--ex_name', name, '--val_batch_size', '64', '--num_workers', '0'])


def collect(name):
    src = ROOT / 'work_dirs' / name
    dest = ROOT / 'results' / name
    dest.mkdir(parents=True, exist_ok=True)
    for path in src.glob('train_*.log'):
        shutil.copy2(path, dest / path.name)
    for path in (src / 'metrics').glob('version_*/metrics.csv'):
        shutil.copy2(path, dest / f'{path.parent.name}_metrics.csv')
    for path in [src / 'saved' / 'metrics.npy', src / 'checkpoints' / 'best.ckpt']:
        shutil.copy2(path, dest / path.name)
    shutil.copytree(src / 'demo', dest / 'demo', dirs_exist_ok=True)
    for suffix in ['', '_gifs', '_predict']:
        path = OUT / f'{name}{suffix}.txt'
        shutil.copy2(path, dest / path.name)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait-for-smoke-pid', type=int)
    args = parser.parse_args()
    os.chdir(ROOT)
    OUT.mkdir(parents=True, exist_ok=True)
    kernel = ctypes.windll.kernel32
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.GetExitCodeProcess.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulong)]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    # Keep the system awake only while this runner is active; restore on exit.
    kernel.SetThreadExecutionState(0x80000001)
    try:
        if args.wait_for_smoke_pid:
            handle = kernel.OpenProcess(0x00100000 | 0x1000, False, args.wait_for_smoke_pid)
            if not handle:
                raise RuntimeError('Cannot attach to smoke process; inspect smoke result before continuing')
            status('Waiting for the current smoke run to finish')
            kernel.WaitForSingleObject(handle, 0xFFFFFFFF)
            code = ctypes.c_ulong()
            kernel.GetExitCodeProcess(handle, ctypes.byref(code))
            kernel.CloseHandle(handle)
            if code.value:
                raise RuntimeError(f'Smoke run failed with exit code {code.value}')
        else:
            train('smoke_xs', 'configs/mmnist/STLight_XS_smoke.py')
        if not (ROOT / 'work_dirs/smoke_xs/saved/metrics.npy').exists():
            raise RuntimeError('Smoke test metrics missing; full runs will not start')
        for variant in ['xs', 's']:
            name = f'mmnist_stlight_{variant}'
            train(name, f'configs/mmnist/STLight_{variant.upper()}.py')
            run(name + '_gifs', ['tools/demo_gifs.py', '--res_dir', f'work_dirs/{name}', '--n_samples', '3'])
            run(name + '_predict', ['tools/demo_predict.py', '--exp_dir', f'work_dirs/{name}', '--n_samples', '3'])
            collect(name)
        status('Training and artifact collection complete; results ready for review and Git push')
    except Exception as exc:
        status(f'STOPPED: {exc}')
        raise
    finally:
        kernel.SetThreadExecutionState(0x80000000)


if __name__ == '__main__':
    main()
