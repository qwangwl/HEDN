import os


# ablations = [
#     "main",
#     "abl_sra_random",
#     "abl_sra_w_hard",
#     "abl_sra_w_easy",
#     "abl_comp_wo_easy",
#     "abl_comp_wo_hard",
#     "abl_comp_wo_clusterloss",
#     "abl_comp_wo_clusterloss_target",
#     "abl_comp_wo_clusterloss_source",
#     "abl_comp_two_stage",
# ]

ablations = ["abl_comp_wo_hard"]
dataset_name = "seed3"
for ablation in ablations:
    saved_model = True if ablation == "main" else False
    early_stop = 0 if ablation == "main" else 1000
    for sess in [1, 2, 3]:
        command = (
            f"python cross_subject.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--ablation {ablation} "
            f"--saved_model {saved_model} "
            f"--early_stop {early_stop} "
            f"--tmp_saved_path logs//ablation//{ablation}// "
        )
        os.system(command)

ablations = [
    "main",
    "abl_sra_random",
    "abl_sra_w_hard",
    "abl_sra_w_easy",
    "abl_comp_wo_easy",
    "abl_comp_wo_hard",
    "abl_comp_wo_clusterloss",
    "abl_comp_wo_clusterloss_target",
    "abl_comp_wo_clusterloss_source",
    "abl_comp_two_stage",
]
dataset_name = "seed4"
for ablation in ablations:
    saved_model = True if ablation == "main" else False
    early_stop = 0 if ablation == "main" else 1000
    for sess in [1, 2, 3]:
        command = (
            f"python cross_subject.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--ablation {ablation} "
            f"--saved_model {saved_model} "
            f"--early_stop {early_stop} "
            f"--tmp_saved_path logs//ablation//{ablation}// "
        )
        os.system(command)


dataset_name = "deap"
for emotion in ['valence', 'arousal']:
    command = (
        f"python cross_subject.py "
        f"--dataset_name {dataset_name} "
        f"--emotion {emotion} "
        f"--saved_model {saved_model} "
        f"--tmp_saved_path logs//ablation//{ablation}// "
    )
    os.system(command)


dataset_name = "seed3"
abl = "num_sources"
for num_sources in [14, 12, 10, 8, 6, 4, 2]:
    for sess in [1, 2, 3]:
        tmp_saved_path = f"logs//num_sources//{num_sources}"
        command = (
            f"python cross_subject.py "
            f"--dataset_name {dataset_name} "
            f"--session {sess} "
            f"--ablation {abl} "
            f"--num_sources {num_sources} "
            f"--tmp_saved_path {tmp_saved_path} "
        )
        os.system(command)