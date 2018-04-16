import pickle
import gzip
import numpy
import csv
import pandas as pd
import matplotlib.pyplot as plt
from pylab import *

inputNeuron, hiddenNeuron, outputNeuron = input("Enter size for each layer: ").split()
# seed (initial value) random number generator to help train the network
numpy.random.seed(2000)
# load training and test data
with gzip.open("TrainDigitX.csv.gz", 'r') as f:
    training = csv.reader(f)
with open("TestDigitX.csv.gz", 'r', encoding='latin1') as f:
    testing = csv.reader(f)
    data = pd.read_csv('TrainDigitX.csv')

# load validation data by splitting

''' dataset restructure, inputs to the neural network are going to be tuples (x, y) x is input and y is correct output
    a number between 1-9 (output) needs to be encoded, this is done by using 10 output neurons
    neuron with best activation will be the predication for the network
    output y has to be the list of 10 numbers, all are 0 expect for the correct digits (entry)'''
validate_set = [(validation[0][i], [1 if j == validation[1][i] else 0 for j in range(10)])
                    for i in numpy.arange(len(validation[0]))]
train_set = [(training[0][i], [1 if j == training[1][i] else 0 for j in range(10)])
                    for i in numpy.arange(len(training[0]))]
test_set = [(testing[0][i], [1 if j == testing[1][i] else 0 for j in range(10)])
                    for i in numpy.arange(len(testing[0]))]


# sigmoid function and its derivative
def s(x):
    return 1 / (1 + numpy.exp(-x))
def derivative_s(x):
    return s(x) * (1 - s(x))

class entropy:
    def fn(a, t): # activation/target
        return -(t * numpy.log(a) + (1 - t) * numpy.log(1 - a))

    def fn_deriv(a, t):
        # derivative of the cross entropy
        return -(t / a - (1 - t) / (1 - a))

    def delta(intake, a, t):
        # delta error at output layer (cross entropy cost)
        return (a - t)

class quadratic:
    def fn(a, t):
        # Evaluate
        return 0.5 * (a - t) ** 2

    def fn_deriv(a, t):
        # Evaluate derivative
        return a - t

    def delta(intake, a, t):
        # delta error at output layer (quadratic cost)
        return (a - t) * derivative_s(intake)


class neuralnetwork:
        def __init__(network, body, cost=quadratic): # either equal to cost=quadratic or cost=entropy
            network.cost = cost  # cost function
            network.body = body # body of network is stored
            network.layers = len(body) # amount of layers

            #  initialising the biases for the hidden/output layers
            network.b = [numpy.random.normal(0, 1, (body[i])) for i in range(1, network.layers)]
            # initialising the weight matrices, rescaled for peaked activation
            network.w = [numpy.random.normal(0, 1 / numpy.sqrt(body[i + 1]), (body[i], body[i + 1])) for i in range(network.layers - 1)]

        def backpropagate(network, t):
            # propagate the error backwards
            network.dcdw = {} # dcdw = delta cost delta weight
            network.d = {} # delta
            network.dcdb = {} # dcdb =delta cost delta bias

            # Delta for final output
            network.d[network.layers - 1] = (network.cost).delta(network.input[network.layers - 1], network.output[network.layers - 1], t)
            # deltas for the other layers
            for l in numpy.arange(network.layers - 2, -1, -1):
                network.d[l] = numpy.dot(network.d[l + 1], network.w[l].T) * derivative_s(numpy.array(network.input[l]))

            # partial derivatives - bias and weight
            for l in numpy.arange(network.layers - 1, 0, -1):
                network.dcdb[l] = network.d[l]
                network.dcdw[l] = numpy.dot(network.output[l - 1].T, network.d[l])
            return network.dcdb, network.dcdw

        def forwardinput(network, inData):
            # need to store input/outputs for each layer
            network.output = {}
            network.input = {}
            # no activation function for input
            network.input[0] = inData
            network.output[0] = numpy.array(inData)

            # Feed input data into network
            for l in range(1, network.layers):
                network.input[l] = numpy.dot(network.output[l - 1], network.w[l - 1]) + network.b[l - 1]
                network.output[l] = numpy.array(s(network.input[l]))
            # return output (last layer)
            return network.output[network.layers - 1]

        def batch_train(network, dt, r, reg):
            # train on a mini batch
            # Split into input/output
            intake = [entered[0] for entered in dt] # input through network
            t = [entered[1] for entered in dt] # target
            # Feed input
            network.forwardinput(intake)
            # propagate error backwards
            network.backpropagate(t)
            # Update bias/weights
            n = len(t)
            for l in numpy.arange(1, network.layers):
                network.b[l - 1] -= (r) * numpy.mean(network.dcdb[l], axis=0)
                network.w[l - 1] -= (r / n) * network.dcdw[l] - r * reg * network.w[l - 1]

        def evaluate(network, test_set):
            # performance test by counting examples in test sample that were correctly evaluated
            count = 0
            for i in test_set:
                p = numpy.argmax(network.forwardinput(i[0]))
                out = numpy.argmax(i[1])
                count = count + 1 if (out - p) == 0 else count
            return count

        def SGD(network, dt, epochs, batchSize, r=1, reg=0.1, test_set=None):
            # Train using stochastic gradient descent
            for e in numpy.arange(epochs):
                # randomly split dt (data) into epochs
                numpy.random.shuffle(dt)
                bts = [dt[x:x + batchSize] for x in numpy.arange(0, len(dt), batchSize)]
                for bt in bts:
                    network.batch_train(bt, r, reg)
                if dt != None:
                    # show results of test
                    print(
                        "Test Epoch {0} got {1} out of {2}".format(e + 1, network.evaluate(test_set), len(test_set)))

            # Set biases/weights
            network.w = dt['w']
            network.b = dt['b']
            print("Weights: {0} \nBias is: {1}".format(dt['b'], dt['w']))
            # The following creates a neural network of 784 input neurons, 30 hidden neurons and 10 output neurons
			# can substitute values from user input
            n = neuralnetwork([784, 30, 10])
            # using SGD, train using epoch=30, mini-batch size=20, learning rate=3.0 (default) regularization parameter=0.001
            n.SGD(train_set, 30, 20, 3, 0.001 / len(training[0]), test_set=validate_set)
            # go through the entire data set and try to predict each number
        for i in range(0, len(test_set)):
            # choose random to test
            fileImgNum = numpy.random.randint(0, 10000)
            # feed it through network to get prediction (p)
            p = n.forwardinput(testing[0][fileImgNum])
            print("Image number: {0} \nActual Number: {1} \nNetwork Predicition: {2}".format(fileImgNum, testing[1][fileImgNum], numpy.argmax(p)))
            # produce message if successful or not
            if testing[1][fileImgNum] == numpy.argmax(p):
                print("Success! Network predicted the right number!")
            else:
                print("Network predicted the wrong number!")
        '''
          *COMMENTED OUT* - Used to save the results to the prediction files
            np.savetxt('PredictDigitY1.csv',
            np.c_[range(1,len(test_set)+1), numpy.argmax(p)],
            delimiter=',',
            fmt='%d')'''