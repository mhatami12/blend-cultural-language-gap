#!/usr/bin/env bash
# One-time environment setup.
set -euo pipefail
pip install -r requirements.txt
pip install --no-deps hazm==0.10.0
python -m spacy download en_core_web_sm
python -c "from hazm import Lemmatizer; assert Lemmatizer().lemmatize('کارمندی') == 'کارمندی'; print('hazm OK')"
