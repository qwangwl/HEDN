import os

dataset_name ="seed3"
for sess in [1, 2, 3]:
    command = (
        f"python cross_subject.py "
        f"--dataset_name {dataset_name} "
        f"--session {sess} "
    )
    os.system(command)

dataset_name = "seed4"
for sess in [1, 2, 3]:
    command = (
        f"python cross_subject.py "
        f"--dataset_name {dataset_name} "
        f"--session {sess} "
    )
    os.system(command)

dataset_name = "deap"
for emotion in ['valence', 'arousal']:
    command = (
        f"python cross_subject.py "
        f"--dataset_name {dataset_name} "
        f"--emotion {emotion} "
    )
    os.system(command)