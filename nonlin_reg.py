import numpy as np
x = np.array([[1,0,1], [0,1,0], [0,0,1], [1,0,0]])
y = np.array([[0,1,1,0]]).T
np.random.seed(0)
weights = np.random.random((3,1))

for iteration in range(100000):
	z = np.dot(x, weights)
	sigmoid = 1/(1+np.exp(-z))
	error = (y - sigmoid)
	sigmoidDerivative= sigmoid * (1 - sigmoid)
	# sigmoidDerivative = sigmoid * (1 - sigmoid)
	weights += np.dot(x.T, error*sigmoidDerivative)

	newZ = np.dot(np.array([0,1,1]), weights)
	activationOutput = 1/(1+np.exp(-newZ))

	if (abs(1-activationOutput) < 0.001):
		break
	print iteration, abs(1-activationOutput)



print("Considering new situation: [0,1,1]")
# calculate weighted inputs
newZ = np.dot(np.array([0,1,1]), weights)
# put weighted inputs into our activation function to get the network's output
activationOutput = 1/(1+np.exp(-newZ))
print(activationOutput)