import os

# all parameters are set in transfer.yaml
config_path = "configs\\transfer.yaml"
seed = 42

dataset_name = "seed3"
abl = "num_sources"
for num_sources in [14, 12, 10, 8, 6, 4, 2]:
    for sess in [3]:
        tmp_saved_path = f".\\logs_num_sources\\{num_sources}"
        command = (
            f"python main.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--ablation {abl} "
            f"--num_sources {num_sources} "
            f"--tmp_saved_path {tmp_saved_path} "
        )
        os.system(command)