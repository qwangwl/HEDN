import os

for seed in [40, 41, 43, 44]:
    cammand = (
        f"python cross_dataset.py "
        f"--seed {seed}"
    )
    os.system(cammand)