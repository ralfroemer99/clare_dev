STEPS=30000;
LOG_STEPS=100;
BS=128;

python ./lerobot_lsy/src/lerobot/scripts/train_real.py \
    --seed=42 \
    --job_name=scratch_franka_task_0 \
    --output_dir=./outputs/real/scratch/scratch_franka_task_0 \
    --dataset.repo_id=continuallearning/real_0_put_bowl_filtered \
    --dataset.video_backend=pyav \
    --policy.type=ditflow_mt \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=16 \
    --steps=$STEPS \
    --eval_freq=0 \
    --save_freq=10000 \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_experiments \
    --wandb.entity=ralf-roemer \
&& \
python ./lerobot_lsy/src/lerobot/scripts/train_real.py \
    --seed=42 \
    --job_name=scratch_franka_task_1 \
    --output_dir=./outputs/real/scratch/scratch_franka_task_1 \
    --dataset.repo_id=continuallearning/real_1_stack_bowls_filtered \
    --dataset.video_backend=pyav \
    --policy.type=ditflow_mt \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=16 \
    --steps=$STEPS \
    --eval_freq=0 \
    --save_freq=10000 \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_experiments \
    --wandb.entity=ralf-roemer \
&& \
python ./lerobot_lsy/src/lerobot/scripts/train_real.py \
    --seed=42 \
    --job_name=scratch_franka_task_2 \
    --output_dir=./outputs/real/scratch/scratch_franka_task_2 \
    --dataset.repo_id=continuallearning/real_2_put_moka_pot_filtered \
    --dataset.video_backend=pyav \
    --policy.type=ditflow_mt \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=16 \
    --steps=$STEPS \
    --eval_freq=0 \
    --save_freq=10000 \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_experiments \
    --wandb.entity=ralf-roemer \
&& \
python ./lerobot_lsy/src/lerobot/scripts/train_real.py \
    --seed=42 \
    --job_name=scratch_franka_task_3 \
    --output_dir=./outputs/real/scratch/scratch_franka_task_3 \
    --dataset.repo_id=continuallearning/real_3_close_drawer_filtered \
    --dataset.video_backend=pyav \
    --policy.type=ditflow_mt \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=16 \
    --steps=$STEPS \
    --eval_freq=0 \
    --save_freq=10000 \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_experiments \
    --wandb.entity=ralf-roemer \
&& \
python ./lerobot_lsy/src/lerobot/scripts/train_real.py \
    --seed=42 \
    --job_name=scratch_franka_task_4 \
    --output_dir=./outputs/real/scratch/scratch_franka_task_4 \
    --dataset.repo_id=continuallearning/real_4_put_lego_into_drawer_filtered \
    --dataset.video_backend=pyav \
    --policy.type=ditflow_mt \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=16 \
    --steps=$STEPS \
    --eval_freq=0 \
    --save_freq=10000 \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_experiments \
    --wandb.entity=ralf-roemer;
