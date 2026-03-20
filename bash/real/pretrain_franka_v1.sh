STEPS=200000;
LOG_STEPS=100;
BS=256;

python ./lerobot_lsy/src/lerobot/scripts/train_real.py \
    --seed=42 \
    --job_name=dit_flow_mt_franka_pretrain__v1_B$BS \
    --output_dir=./outputs/real/dit_flow_mt_franka_pretrain_v1_B$BS \
    --dataset.repo_id=continuallearning/pretraining_v1_lerobot30 \
    --dataset.video_backend=pyav \
    --policy.type=ditflow_mt \
    --policy.push_to_hub=false \
    --batch_size=$BS \
    --num_workers=16 \
    --steps=$STEPS \
    --eval_freq=0 \
    --save_freq=50000 \
    --log_freq=$LOG_STEPS \
    --wandb.enable=true \
    --wandb.disable_artifact=true \
    --wandb.project=clare_experiments \
    --wandb.entity=ralf-roemer;
