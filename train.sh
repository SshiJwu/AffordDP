DEBUG=True
save_ckpt=True

seed=${1}
cuda_id=${2}

if [ $DEBUG = True ]; then
    wandb_mode=offline
    # wandb_mode=online
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
    echo -e "\033[33mDebug mode!\033[0m"
else
    wandb_mode=online
    echo -e "\033[33mTrain mode\033[0m"
fi


python train_policy.py --config-name=afford_cond_pointcloud_dp.yaml \
                        task=PullDrawer \
                        training.seed=${seed} \
                        training.device="cuda:${cuda_id}" \
                        logging.mode=${wandb_mode} \
                        checkpoint.save_ckpt=${save_ckpt}
