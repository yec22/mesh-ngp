#! /bin/bash
OMP_NUM_THREADS=8 CUDA_VISIBLE_DEVICES=1 python main_nerf.py data/neus/dtu_scan105 --workspace exps/trial_dtu_scan105 -O --bound 1 --scale 1.0 --dt_gamma 0 --iters 15000