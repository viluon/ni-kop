#!/bin/bash

set -euxo pipefail

python3 -m venv entangled-filters
source entangled-filters/bin/activate
pip3 install wheel
pip3 install wheels/*
pip3 install -r requirements.txt
patch entangled-filters/lib/python*/site-packages/entangled/doctest.py < doctest.patch
cat entangled-filters/bin/pandoc-bootstrap
mkdir -p docs
touch docs/bench.csv
make site
