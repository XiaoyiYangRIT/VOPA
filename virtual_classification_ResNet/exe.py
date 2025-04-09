import subprocess

selector = 8
# Run the other script
#for i in range(10):
#    subprocess.run(["python", "mixed_training_cv.py", str(i)])
subprocess.run(["python", "mixed_training_cv.py", str(selector)])