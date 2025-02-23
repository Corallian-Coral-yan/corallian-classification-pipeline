# How to run on COARE
1. Clone the repo on COARE
2. Upload/move the already cropped dataset and the generated annotations file (index.csv) to a location on the scratch folder
3. Edit `config.toml`
4. Create the environment with `sbatch create-env.slurm`
5. Run `train.py` with `sbatch run-train.slurm`. The output will be in `run-train.out` once the model is finished training
