#!/bin/bash

set -euxo pipefail

python3 -m venv entangled-filters
source entangled-filters/bin/activate
pip3 install --use-deprecated=legacy-resolver wheels/*
pip3 install --use-deprecated=legacy-resolver -r requirements.txt
python3 -m zsh_jupyter_kernel.install
patch entangled-filters/lib/python*/site-packages/entangled/doctest.py < doctest.patch
cat entangled-filters/bin/pandoc-bootstrap
mkdir -p docs
touch docs/bench.csv
make all
