__author__ = 'Marco Giancarli -- m.a.giancarli@gmail.com'


from sklearn import preprocessing as prep
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.metrics import roc_auc_score
from sklearn.svm import SVR

from pybrain.structure import FeedForwardNetwork
from pybrain.structure import LinearLayer, SigmoidLayer, BiasUnit
from pybrain.structure import FullConnection
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import ClassificationDataSet
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.xml.networkwriter import NetworkWriter

import math
import numpy as np
import csv


if __name__ == '__main__':
    train_bidders_list = []
    test_bidders_list = []
    outcomes_list = []

    train_bidder_ids = []
    test_bidder_ids = []

    OVERSAMPLE_VALUE = 10 # copy all of the bots in test data this many times
    BID_THRESHOLD = 1  # bidders must have at least this many bids
    bad_examples = {}  # not enough data to be relevant
    bot_examples_indices = []  # track the indices of bots for oversampling

    with open('../gen/train.csv') as train_file:
        for line in train_file.readlines()[1:]:
            line = line.split('\n')[0]
            read_values = line.split(',')
            used_values = ()

            for value in read_values[3:]:
                number = float(value)
                # add the inverse of the values or 10e3 if divided by zero
                inverse = math.pow(number, -1) if math.fabs(number) > 1e-5 \
                                               else math.copysign(1e5, number)
                used_values += (number, inverse)
            train_bidders_list.append(used_values)
            train_bidder_ids.append(read_values[0])

    with open('../gen/outcomes.csv') as outcomes_file:
        for line in outcomes_file.readlines()[1:]:
            line = line.split('\n')[0]
            outcomes_list.append(float(line))

    with open('../gen/test.csv') as test_file:
        for line in test_file.readlines()[1:]:
            line = line.split('\n')[0]
            read_values = line.split(',')
            used_values = ()

            for value in read_values[3:]:
                number = float(value)
                # add the inverse of the values or 10e3 if divided by zero
                inverse = math.pow(number, -1) if math.fabs(number) > 1e-5 \
                                               else math.copysign(1e5, number)
                used_values += (number, inverse)
            test_bidders_list.append(used_values)
            test_bidder_ids.append(read_values[0])


    # remove samples where the number of bids is less than the threshold
    new_train_bidders_list = []
    new_outcomes_list = []
    new_train_bidder_ids = []
    for index in range(len(train_bidders_list)):
        if train_bidders_list[index][0] >= BID_THRESHOLD:
            new_train_bidders_list.append(train_bidders_list[index])
            new_outcomes_list.append(outcomes_list[index])
            new_train_bidder_ids.append(train_bidder_ids[index])
    train_bidders_list = new_train_bidders_list
    outcomes_list = new_outcomes_list
    train_bidder_ids = new_train_bidder_ids

    # oversampling (in place)
    for index in range(len(outcomes_list)):
        if outcomes_list[index] == 1.0:
            bot_examples_indices.append(index)
    for index in range(len(bot_examples_indices)):
        bot_examples_indices[index] += index*OVERSAMPLE_VALUE
    for index_of_bot in bot_examples_indices:
        for dummy in range(OVERSAMPLE_VALUE):
            train_bidders_list.insert(index_of_bot + 1,
                                      train_bidders_list[index_of_bot])
            outcomes_list.insert(index_of_bot + 1,
                                 outcomes_list[index_of_bot])
            train_bidder_ids.insert(index_of_bot + 1,
                                    train_bidder_ids[index_of_bot])


    # set up polynomial feature generator
    poly = prep.PolynomialFeatures(3)

    # make arrays with the training data and scale X
    train_y = np.array(outcomes_list)
    train_X = np.array(train_bidders_list)
    train_X = poly.fit_transform(train_X)
    train_X = prep.scale(train_X)

    # set up feature selection
    selector = SelectKBest(f_regression, k=5)
    selector.fit(train_X, train_y)
    print selector.scores_
    print selector.get_support(indices=True)
    train_X = selector.transform(train_X)

    # get num inputs from the first example in train_X
    num_inputs = len(train_X[0])

    # set up pybrain dataset
    training_set = ClassificationDataSet(num_inputs, nb_classes=1)
    for X,y in zip(train_X, train_y):
        training_set.addSample(X, y)


    # create model
    model = FeedForwardNetwork()

    # create layers
    input_layer = LinearLayer(num_inputs)
    input_bias = BiasUnit()
    hidden_layer = LinearLayer(50)
    hidden_bias = BiasUnit()
    output_layer = SigmoidLayer(1)

    # add layers to model
    model.addInputModule(input_layer)
    model.addModule(input_bias)
    model.addModule(hidden_layer)
    model.addModule(hidden_bias)
    model.addOutputModule(output_layer)

    # create connections
    input_hidden_connection = FullConnection(input_layer, hidden_layer)
    input_hidden_bias_connection = FullConnection(input_bias, output_layer)
    hidden_output_connection = FullConnection(hidden_layer, output_layer)
    hidden_output_bias_connection = FullConnection(hidden_bias, output_layer)

    # add connections
    model.addConnection(input_hidden_connection)
    model.addConnection(hidden_output_connection)

    # sort the shit
    model.sortModules()

    # train the thing - train until convergence in testing to use validation set
    trainer = BackpropTrainer(model, dataset=training_set, momentum=0.2,
                              verbose=True, weightdecay=0.01)
    # trainer.trainEpochs(40)
    trainer.trainUntilConvergence(maxEpochs=60)

    # back it up for later
    NetworkWriter.writeToFile(model, '../gen/network_data')

    # use this to recover a network
    # model = NetworkReader.readFrom('../gen/network_data')

    # make arrays with the test data and scale X
    test_X = np.array(test_bidders_list)
    test_X = poly.fit_transform(test_X)
    test_X = prep.scale(test_X)
    test_X = selector.transform(test_X)

    # predict for test data
    test_y = [model.activate(X)[0] for X in test_X]
    print 'First 50 predictions:'
    print ['%1.5f' % y for y in test_y[:10]]
    print ['%1.5f' % y for y in test_y[10:20]]
    print ['%1.5f' % y for y in test_y[20:30]]
    print ['%1.5f' % y for y in test_y[30:40]]
    print ['%1.5f' % y for y in test_y[40:50]]

    # throw it in a csv file
    with open('../gen/submission.csv', 'w') as output_file:
        output_writer = csv.writer(output_file)
        titles = ('bidder_id','prediction')
        output_writer.writerow(titles)
        for prediction,bidder_id in zip(test_y,test_bidder_ids):
            output_writer.writerow([bidder_id, prediction])