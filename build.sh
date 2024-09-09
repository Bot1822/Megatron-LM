#!/bin/bash

DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
python3 setup_br.py sdist bdist_wheel --dist-dir=${DIR}/whls/