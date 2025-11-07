sbatch run_ablations.sh
sbatch run_ablations.sh --swiglu
sbatch run_ablations.sh --rotary
sbatch run_ablations.sh --lr-linear
sbatch run_ablations.sh --lr-cosine
sbatch run_ablations.sh --rmsnorm
sbatch run_ablations.sh --swiglu --rotary --lr-cosine --rmsnorm