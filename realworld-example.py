


import numpy as np

from perceptron import Perceptron



training_inputs = []
training_inputs.append(np.array([1,0,0]))
training_inputs.append(np.array([0,1,1]))
training_inputs.append(np.array([1,1,1]))

# The Output of the given training data

labels = np.array([1,1,0])

percept = Perceptron(3)
percept.train(training_inputs, labels)

inputs = np.array([0,1,0])
print(percept.predict(inputs))

inputs = np.array([0,0,1])
print(percept.predict(inputs))

inputs = np.array([0,0,1])
print(percept.predict(inputs))

inputs = np.array([1,0,1])
print(percept.predict(inputs))
