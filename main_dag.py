
import sys
import os
import subprocess

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

if __name__ == "__main__":
    cmd = [
        sys.executable, 
        os.path.join(os.path.dirname(__file__), "src\\tpcm_generator\\bin\\dag.py"),
    ]
    for parameter in sys.argv[1:]:
        cmd.append(parameter)
    subprocess.run(cmd, check=True)