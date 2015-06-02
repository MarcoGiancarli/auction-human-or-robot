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


def make_network(num_inputs, num_hidden=100):
    # create model
    model = FeedForwardNetwork()

    # create layers
    input_layer = LinearLayer(num_inputs, name='input-layer')
    input_bias = BiasUnit(name='input-bias')
    hidden_layer = LinearLayer(num_hidden, name='hidden-layer')
    hidden_bias = BiasUnit(name='hidden-bias')
    output_layer = SigmoidLayer(1, name='output-layer')

    # add layers to model
    model.addInputModule(input_layer)
    model.addModule(input_bias)
    model.addModule(hidden_layer)
    model.addModule(hidden_bias)
    model.addOutputModule(output_layer)

    # create connections
    input_hidden_connection = FullConnection(input_layer, hidden_layer, name='input-to-hidden')
    input_hidden_bias_connection = FullConnection(input_bias, output_layer, name='input-bias-to-hidden')
    hidden_output_connection = FullConnection(hidden_layer, output_layer, name='hidden-to-output')
    hidden_output_bias_connection = FullConnection(hidden_bias, output_layer, name='hidden-bias-to-output')

    # add connections
    model.addConnection(input_hidden_connection)
    model.addConnection(input_hidden_bias_connection)
    model.addConnection(hidden_output_connection)
    model.addConnection(hidden_output_bias_connection)

    # sort the shit
    model.sortModules()

    return model


if __name__ == '__main__':
    train_bidders_list = []
    test_bidders_list = []
    outcomes_list = []

    train_bidder_ids = []
    test_bidder_ids = []

    OVERSAMPLE_VALUE = 10 # copy all of the bots in test data this many times
    BID_THRESHOLD = 2  # bidders must have at least this many bids
    bad_examples = {}  # not enough data to be relevant
    bot_examples_indices = []  # track the indices of bots for oversampling

    with open('../gen/train.csv') as train_file:
        for line in train_file.readlines()[1:]:
            line = line.split('\n')[0]
            read_values = line.split(',')
            used_values = ()

            for value in read_values[3:]:
                number = float(value)
                # add the log of the values plus one
                log = math.log(number + 1)
                # add the inverse of the values or 10e3 if divided by zero
                inverse = math.pow(number, -1) if math.fabs(number) > 1e-5 \
                                               else math.copysign(1e5, number)
                # add the inverse of the log of the values plus one
                inverse_log = math.pow(log, -1) if log > 1e-5 else 1e5
                used_values += (number, log, inverse, inverse_log)
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
                # add the log of the values plus one
                log = math.log(number + 1)
                # add the inverse of the values or 10e3 if divided by zero
                inverse = math.pow(number, -1) if math.fabs(number) > 1e-5 \
                                               else math.copysign(1e5, number)
                # add the inverse of the log of the values plus one
                inverse_log = math.pow(log, -1) if log > 1e-5 else 1e5
                used_values += (number, log, inverse, inverse_log)
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
    selector = SelectKBest(f_regression, k='all')
    selector.fit(train_X, train_y)
    # print selector.get_support(indices=True)
    train_X = selector.transform(train_X)

    # get num inputs from the first example in train_X
    num_inputs = len(train_X[0])

    # set up pybrain dataset
    training_set = ClassificationDataSet(num_inputs, nb_classes=1)
    for X,y in zip(train_X[400:], train_y[400:]):  ######################################
        training_set.addSample(X, y)


    # create model
    hidden_layer_size = 800
    num_epochs = 15
    models = [make_network(num_inputs, hidden_layer_size) for dummy in range(5)]
    network_numbers = range(1, len(models) + 1)
    # network_numbers = [1,2,3,6,7,8,10,13,15,18,20,21,22]

    # train the thing - train until convergence in testing to use validation set
    for model_index in range(len(models)):
        trainer = BackpropTrainer(models[model_index], dataset=training_set,
                                  momentum=0.2, verbose=True, weightdecay=0.02)
        for dummy in range(num_epochs):
            trainer.trainEpochs(1)
            temp_predictions = [0 for dummy in range(400)]
            for sample_index in range(400):
                temp_predictions[sample_index] += models[model_index].activate(train_X[sample_index])
            score = roc_auc_score(train_y[:400], temp_predictions)
            print score
            # if score > 0.91:
            #     NetworkWriter.writeToFile(models[model_index], '../gen/network_data' + str(100 + dummy))


        # trainer.trainUntilConvergence(maxEpochs=60)

        # back it up for later
        # NetworkWriter.writeToFile(models[model_index], '../gen/network_data' + str(network_numbers[model_index]))

        # use this to recover a network
        # models[model_index] = NetworkReader.readFrom('../gen/network_data' + (index + 1))

    # make arrays with the test data and scale X
    test_X = np.array(test_bidders_list)
    test_X = poly.transform(test_X)
    test_X = prep.scale(test_X)
    test_X = selector.transform(test_X)

    # predict for test data
    test_y = [0 for dummy in range(len(test_X))]
    for index in range(len(test_X)):
        for model in models:
            test_y[index] += model.activate(test_X[index])
        test_y[index] /= len(models)

    # get ROC AUC for first 400 training samples
    print ''
    nerd = 400
    predictions = [0 for dummy in range(nerd)]
    for model in models:
        temp_predictions = [0 for dummy in range(nerd)]
        for index in range(nerd):
            temp_predictions[index] += model.activate(train_X[index])
        print roc_auc_score(train_y[:nerd], temp_predictions)
        for index in range(nerd):
            predictions[index] += temp_predictions[index]
    predictions = [p / len(models) for p in predictions]
    print '----------'
    print roc_auc_score(train_y[:nerd], predictions)

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