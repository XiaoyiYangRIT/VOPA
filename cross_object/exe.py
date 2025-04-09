import subprocess
import time


def main():
    # Run the other script
    for i in range(10):
        for j in range(3):
            try:
                # Run the subprocess with a 40-minute timeout
                subprocess.run(["python", "mixed_training_cv.py", str(i)], timeout=3000)
            except Exception as e:
                print(e)
            except subprocess.TimeoutExpired:
                print("The subprocess timed out. Starting the next one.")
    
        
if __name__ == "__main__":
    main()
    