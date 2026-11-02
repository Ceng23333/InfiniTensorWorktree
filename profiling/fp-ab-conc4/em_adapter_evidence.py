"""Run the existing adapter unchanged and retain loaded-library provenance."""
import atexit
import hashlib
import json
from pathlib import Path
import runpy
import sys

script = sys.argv.pop(1)
sys.argv[0] = script
output = Path(sys.argv[sys.argv.index('--run-dir') + 1])
case = Path(sys.argv[sys.argv.index('--case-json') + 1]).stem


def evidence():
    maps = Path('/proc/self/maps').read_text()
    paths = {line.split()[-1] for line in maps.splitlines()
             if any(name in line for name in ('libinfinicore_runtime.so', 'libinfiniops.so', '_infinilm.', '_infinicore.'))}
    hashes = {path: hashlib.sha256(Path(path).read_bytes()).hexdigest() for path in paths if Path(path).is_file()}
    directory = output / 'maps'
    directory.mkdir(exist_ok=True)
    (directory / f'{case}.json').write_text(json.dumps(hashes, indent=2) + '\n')


atexit.register(evidence)
runpy.run_path(script, run_name='__main__')
