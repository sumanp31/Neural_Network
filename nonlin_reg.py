import numpy as np
x = np.array([[1,0,1], [0,1,0], [0,0,1], [1,0,0]])
y = np.array([[0,1,1,0]]).T
weights = np.random.random((3,1))

for iteration in range(1000):
	z = np.dot(x, weights)
	sigmoid = np.maximum(0,z)
	error = (y - sigmoid)
	sigmoidDerivative= np.zeros(z.shape)
	sigmoidDerivative[z <= 0] = 0
	sigmoidDerivative[z > 0] = 1
	# sigmoidDerivative = sigmoid * (1 - sigmoid)
	weights += np.dot(x.T, error*sigmoidDerivative)

print("Considering new situation: [0,1,1]")
# calculate weighted inputs
newZ = np.dot(np.array([0,1,1]), weights)
# put weighted inputs into our activation function to get the network's output
activationOutput = np.maximum(0,newZ)
print(activationOutput)