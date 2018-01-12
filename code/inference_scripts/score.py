from sklearn.metrics import log_loss
import numpy as np

with open('kaggle_submission.csv', 'r') as f:
  preds = dict()
  f.readline()
  for line in f.readlines():
    split_line = line.split(',')
    pred = float(split_line[1])
    _id, zone = split_line[0].split('_')
    preds[(_id, zone)] = pred

with open('stage1_solution.csv', 'r') as f:
  solutions = dict()
  f.readline()
  for line in f.readlines():
    split_line = line.split(',')
    solution = float(split_line[1])
    _id, zone = split_line[0].split('_')
    solutions[(_id, zone)] = solution

loss = 0
sols, ps = [], []
for key, value in preds.items():
	sols.append([solutions[key]])
	clipped_pred = preds[key]
	clipped_pred = min(max(clipped_pred, 0.01), 0.99)
	# if clipped_pred == 0.004:
	# 	clipped_pred = 0.05
	ps.append([1-clipped_pred, clipped_pred])
loss = log_loss(np.array(sols), np.array(ps))

print(loss)