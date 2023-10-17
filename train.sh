eval "$(conda shell.bash hook)"
sudo rm -r data
conda activate /mnt/6d4d3819-c0f1-4f9d-b056-08610dac2519/CondaEnvs/habitat2
python -m habitat_sim.utils.datasets_download --uids habitat_test_scenes --data-path data/
python -m habitat_sim.utils.datasets_download --uids habitat_test_pointnav_dataset --data-path data/
#python -m habitat_sim.utils.datasets_download --uids mp3d_example_scene --data-path data/
unset HABITAT_ENV_DEBUG
python -u -m habitat_baselines.run --config-name=pointnav/ppo_pointnav_example.yaml
#export HABITAT_ENV_DEBUG=1
#python -u -m habitat_baselines.run --config-name=pointnav/ppo_pointnav_unreal.yaml

