import os
import shutil
import time

SAVE_ITERS = 5000

src = "/path/to/DLC_project_folder/dlc-models/iteration-0/Mice_WellbeingNov18-trainset95shuffle1/train/"
dst = "/path_to_cwd/snapshot_saves/"

highest_iteration = 0

print("Searching highest iteration")
for file in os.listdir(dst):
    if int(file.split(".")[0].split("-")[1]) > highest_iteration:
        highest_iteration = int(file.split(".")[0].split("-")[1])

print(f"New highest iteration: {highest_iteration}")

while True:
    print("Checking for files to save")
    for file in os.listdir(src):
        if file.startswith("snapshot-"):
            if not file in os.listdir(dst):
                print(f"Copying {file}")
                shutil.copy2(os.path.join(src, file), dst)
                if int(file.split(".")[0].split("-")[1]) > highest_iteration:
                    highest_iteration = int(file.split(".")[0].split("-")[1])
                    print(f"New highest iteration: {highest_iteration}")

    for saved_file in os.listdir(dst):
        iteration = int(saved_file.split(".")[0].split("-")[1])
        age = (highest_iteration - iteration) / SAVE_ITERS
        if age > 10 and not iteration % (SAVE_ITERS * 5) == 0:
            print(f"Removing old file {saved_file}")
            os.remove(os.path.join(dst, saved_file))

    time.sleep(240)
