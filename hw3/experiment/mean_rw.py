import sys
import numpy as np

# user parameters
mean_every = 30
fpath = sys.argv[1]

# parse value from file
v = [float(line.split()[-1]) for line in open(fpath, 'r') if 'episode' in line]

# calculate mean value
for i in range(len(v)):
	m = np.array(v[max(i-mean_every,0):i]).astype('float32').mean()
	print('episode: {}   mean_reward(per {}): {}'.format(i, mean_every, m))
