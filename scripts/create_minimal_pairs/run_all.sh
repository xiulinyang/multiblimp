#!/bin/bash

python3 multiblimp/scripts/create_minimal_pairs/sv#a.py --langs "$@"
python3 multiblimp/scripts/create_minimal_pairs/svPa.py --langs "$@"
python3 multiblimp/scripts/create_minimal_pairs/svGa.py --langs "$@"
python3 multiblimp/scripts/create_minimal_pairs/sp#a.py --langs "$@"
python3 multiblimp/scripts/create_minimal_pairs/spPa.py --langs "$@"
python3 multiblimp/scripts/create_minimal_pairs/spGa.py --langs "$@"