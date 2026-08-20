import numpy as np
from  micrograd import Value, MLP
import matplotlib.pyplot as plt
import random

def main():
	np.random.seed(1337)
	random.seed(1337)

	# make up a dataset
	from sklearn.datasets import make_moons, make_blobs
	X, y = make_moons(n_samples=100, noise=0.1)
	y = y*2 - 1 # make y be -1 or 1

	model = MLP(2, [16, 16, 1])
	print(model)
	print(len(model.parameters()))

	for k in range(100):
		inputs = [list(map(Value, xRow)) for xRow in X]
		outputs = list(map(model, inputs))

		loss = [(1 + -yi * si).relu() for yi, si in zip(y, outputs)]
		data_loss = sum(loss) * (1.0 / len(loss))

		alpha = 1e-4
		reg_loss = alpha * sum((p * p for p in model.parameters()))

		total_loss = data_loss + reg_loss

		accuracy_list = [(yi > 0) == (si.data > 0) for yi, si in zip(y, outputs)]
		accuracy = sum(accuracy_list) / len(accuracy_list)

		model.zero_grad()
		total_loss.backward()

		learning_rate = 1.0 - 0.9*k/100
		for p in model.parameters():
			p.data -= learning_rate * p.grad

		if k % 1 == 0:
			print(f"step {k} loss {total_loss.data}, accuracy {accuracy*100}%")

	h = 0.25
	x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
	y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
	xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
						np.arange(y_min, y_max, h))
	Xmesh = np.c_[xx.ravel(), yy.ravel()]
	inputs = [list(map(Value, xrow)) for xrow in Xmesh]
	scores = list(map(model, inputs))
	Z = np.array([s.data > 0 for s in scores])
	Z = Z.reshape(xx.shape)

	fig = plt.figure()
	plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap=plt.cm.Spectral)
	plt.xlim(xx.min(), xx.max())
	plt.ylim(yy.min(), yy.max())
	plt.show()


if __name__ == "__main__":
	main()