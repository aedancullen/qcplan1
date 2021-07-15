
import os
import subprocess

filepath = os.path.abspath(os.path.dirname(__file__))
env = os.environ.copy()
if "LD_LIBRARY_PATH" in env:
    base = env["LD_LIBRARY_PATH"]
else:
    base = ""
env["LD_LIBRARY_PATH"] = base + ":" + filepath + "/ompl/"
p = subprocess.Popen(["python3", filepath + "/core.py"], env=env)
p.communicate()
