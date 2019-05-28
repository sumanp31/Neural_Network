import numpy as np

x = np.array([[1,0,1], [0,1,0], [0,0,1], [1,0,0]])
y = np.array([[0,1,1,0]]).T

m = x.shape[0]
n = x.shape[1]
l = 2
p = 3
np.random.seed(0)
w1 = np.random.random((n,p))
np.random.seed(0)
w2 = np.random.random((p,1))



def actFunc1(z):
	return np.maximum(0,z)

def actFunc2(z):
	return 1/(1+np.exp(-z))

def derActFunc1(z):
	a = np.zeros(z.shape)
	a[z <= 0] = 0
	a[z > 0] = 1
	
	return a

def derActFunc2(z):
	sig = actFunc2(z)
	return sig * (1 - sig)


def forward(x, w1, w2):
	z1 = np.dot(x, w1)
	a1 = actFunc1(z1)
	z2 = np.dot(a1, w2)
	a2 = actFunc2(z2)
	
	return z1, a1, z2, a2

def backward(err, z1, z2, a1, w1, w2):
	dz2 = np.multiply(err, derActFunc1(z2))
	w2 = w2 - (1/float(m)) * np.dot(a1.T, dz2)
	dz1 = np.multiply(np.dot(dz2, w2.T), derActFunc2(z1))
	w1 = w1 - (1/float(m)) * np.dot(x.T, dz1)

	return w1, w2



for i in range(10):
	z1, a1, z2, y_pred = forward(x, w1, w2)
	err = y - y_pred
	w1, w2 = backward(err, z1, z2, a1, w1, w2)


z1, a1, z2, y_test = forward([0,1,1], w1, w2)


print y_test
