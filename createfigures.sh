#!/bin/bash

. ~/.bashrc

mkdir figures 2>/dev/null
python paper_figures_eval.py
python paper_figures_general.py
