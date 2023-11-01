eval "$(conda shell.bash hook)"
conda activate /mnt/6d4d3819-c0f1-4f9d-b056-08610dac2519/CondaEnvs/habitat2
python -u -m habitat_baselines.run --config-name=pointnav/ppo_pointnav_unreal.yaml habitat_baselines.evaluate=True

