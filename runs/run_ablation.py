import os

# command = (
#     "python cross_dataset.py "
# )
# os.system(command)

# command = (
#     "python cross_dataset1.py "
# )
# os.system(command)

ablations = [
    # "main",
    # "abl_tda_random",
    # "abl_tda_wo_easy",
    # "abl_tda_wo_hard",
    # "abl_comp_wo_easy",
    # "abl_comp_wo_hard",
    # "abl_comp_wo_clusterloss",
    "abl_comp_wo_clusterloss_target",
    "abl_comp_wo_clusterloss_source",
    "abl_comp_two_stage",
]

dataset_name = "seed3"
for ablation in ablations:
    saved_model = True if ablation == "main" else False
    early_stop = 0 if ablation == "main" else 1000
    for sess in [1, 2, 3]:
        command = (
            f"python main.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--ablation {ablation} "
            f"--saved_model {saved_model} "
            f"--early_stop {early_stop} "
        )
        os.system(command)


dataset_name = "seed4"
for ablation in ablations:
    saved_model = True if ablation == "main" else False
    early_stop = 0 if ablation == "main" else 1000
    for sess in [1, 2, 3]:
        command = (
            f"python main.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--ablation {ablation} "
            f"--saved_model {saved_model} "
            f"--early_stop {early_stop} "
        )
        os.system(command)