# pi0 HPC quick start

The raw demonstrations and trained outputs are intentionally ignored by Git. Push the code to
GitHub, but transfer data/demos/vla_centered_v4 separately. Dataset conversion and dependency
installation should run before requesting a GPU allocation.

## 1. Clone on the HPC login node

Push this branch to GitHub locally, then run:

~~~bash
ssh HPC_USER@HPC_HOST
cd /scratch/HPC_USER
git clone REPOSITORY_URL robot-pi0
exit
~~~

## 2. Transfer demonstrations, then install

The data directory is ignored by Git. From the local project directory, transfer it into the
repository you just cloned:

~~~bash
rsync -azh --info=progress2 data/demos/vla_centered_v4/ \
  HPC_USER@HPC_HOST:/scratch/HPC_USER/robot-pi0/data/demos/vla_centered_v4/
~~~

Return to the HPC login node and install without consuming GPU allocation time:

~~~bash
ssh HPC_USER@HPC_HOST
cd /scratch/HPC_USER/robot-pi0
bash scripts/hpc/setup_pi0_hpc.sh
source .venv-pi0/bin/activate
~~~

If the cluster does not expose a suitable Python by default, load its Python 3.10-3.12 module
before running setup. PYTHON_BIN=/path/to/python3.12 can also be passed to setup.

## 3. Convert and verify the dataset on the login/CPU node

~~~bash
python scripts/prepare_pi0_dataset.py \
  --demo-dir data/demos/vla_centered_v4 \
  --output-dir data/lerobot/vla_centered_v4 \
  --repo-id local/robot_manipulation_pi0_vla

python scripts/pi0_preflight.py \
  --dataset-dir data/lerobot/vla_centered_v4 \
  --output-dir outputs/pi0_hpc
~~~

Conversion is idempotent. A completed dataset is reused; partial data is rejected.

## 4. Submit training

Add the university-specific partition/account on the command line if required:

~~~bash
sbatch --partition=GPU_PARTITION --account=ACCOUNT scripts/hpc/train_pi0.slurm
squeue -u "$USER"
tail -f pi0-JOB_ID.out
~~~

The job requests one GPU for 1 hour 55 minutes, trains to 3,000 total steps, saves every 500
steps, computes held-out validation loss every 500 steps, and verifies the final checkpoint.
Submitting the same command again resumes from checkpoints/last rather than overwriting it.

Override defaults through exported variables:

~~~bash
sbatch --export=ALL,STEPS=5000,BATCH_SIZE=1,SAVE_FREQUENCY=500 \
  --partition=GPU_PARTITION --account=ACCOUNT scripts/hpc/train_pi0.slurm
~~~

## 5. Validate, then test

~~~bash
sbatch --export=ALL,SPLIT=validation,EPISODES=20 \
  --partition=GPU_PARTITION --account=ACCOUNT scripts/hpc/evaluate_pi0.slurm

sbatch --export=ALL,SPLIT=test,EPISODES=20 \
  --partition=GPU_PARTITION --account=ACCOUNT scripts/hpc/evaluate_pi0.slurm
~~~

Validation uses seeds beginning at 10,000. Test uses a separate range beginning at 20,000.
Metrics are saved to outputs/pi0_hpc/metrics/validation.json and test.json. Do not use test
results to tune training settings.

## 6. Run a language command interactively

~~~bash
source .venv-pi0/bin/activate
MUJOCO_GL=egl python scripts/interactive_pi0_tasks.py \
  --checkpoint outputs/pi0_hpc/checkpoints/last/pretrained_model \
  --device cuda
~~~

The current actuator target is the MuJoCo Franka simulation. Real-robot deployment still requires
a hardware adapter, calibrated cameras, joint limits, emergency stop, and control-rate checks.
