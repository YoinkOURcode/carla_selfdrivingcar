#!/bin/zsh
set -eu
cd -- "${0:A:h}"
if [[ ! -x .venv/bin/python ]]; then
  print 'The local Python environment is missing. See README.md for setup.'
  exit 1
fi
print 'Open http://127.0.0.1:8773 in a browser once the model is ready.'
print 'Keep this terminal open during the demo. Press Control C to stop.'
exec .venv/bin/python demo.py --device mps "$@"
