ckpt=${1}
object_id=${2}
python eval_policy.py --config-name=afford_cond_pointcloud_dp.yaml \
                        task=PullDrawer \
                        hydra.run.dir=${ckpt} \ 
                        task.env_runner.object_id=${object_id}
