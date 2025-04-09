import subprocess

# Run the other script
for i in range(5):
    subprocess.run(["python", "mixed_training_cv.py", str(8)])
#subprocess.run(["python", "mixed_training_cv.py", str(9)])