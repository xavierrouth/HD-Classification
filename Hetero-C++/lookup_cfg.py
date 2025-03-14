import sys

MLC = [1, 2, 3]
WRITE_VERIFY = [1, 2, 3, 4, 5, 6, 7, 8]
LUT = {(1, 1): 0.418858, (1, 2): 0.436818, (1, 3): 0.430404, (1, 4): 0.430404, (1, 5): 0.47338, (1, 6): 0.406029, (1, 7): 0.408595, (1, 8): 0.408595, (2, 1): 0.0295061, (2, 2): 0.0224503, (2, 3): 0.0288647, (2, 4): 0.0256575, (2, 5): 0.0301475, (2, 6): 0.020526, (2, 7): 0.0288647, (2, 8): 0.00833868, (3, 1): 0.0744067, (3, 2): 0.0898012, (3, 3): 0.0801796, (3, 4): 0.0647851, (3, 5): 0.0737652, (3, 6): 0.091084, (3, 7): 0.0673509, (3, 8): 0.0744067}

lower_bound = float(sys.argv[1])
best_mlc = None
best_wv = None
for mlc in MLC:
	for wv in WRITE_VERIFY:
		lut = LUT[(mlc, wv)]
		if lut >= lower_bound:
			if best_mlc is None or best_mlc - best_wv < mlc - wv:
				best_mlc = mlc
				best_wv = wv
print(best_mlc, best_wv)
