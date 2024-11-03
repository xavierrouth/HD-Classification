import subprocess, os

MLC = [1, 2, 3]
WRITE_VERIFY = [1, 2, 3, 4, 5, 6, 7, 8]
myenv = os.environ.copy()
lut = dict()

for mlc in MLC:
	for wv in WRITE_VERIFY:
		myenv["MLC"] = str(mlc)
		myenv["WRITE_VERIFY"] = str(wv)
		subprocess.run(["make", "clean", "host-sim"], env=myenv)
		result = subprocess.run(["./host-sim", "1"], capture_output=True)
		score = float(str(result.stdout).split("Test Accuracy: ")[-1].split("\\")[0])
		lut[(mlc, wv)] = score
		print(lut)
