__author__ = 'Marco Giancarli -- m.a.giancarli@gmail.com'


from sklearn.tree import DecisionTreeRegressor as DTR
from sklearn import preprocessing as prep
import math
import numpy as np
import csv


if __name__ == '__main__':
    train_bidders_list = []
    test_bidders_list = []
    outcomes_list = []

    train_bidder_ids = []
    test_bidder_ids = []

    #TODO: feature scaling

    with open('../gen/train.csv') as train_file:
        for line in train_file.readlines()[1:]:
            line = line.split('\n')[0]
            read_values = line.split(',')
            used_values = ()

            # replace time_stddev with it's log to make the value simpler
            if float(read_values[10]) >= 1.0:
                read_values[10] = str(math.log(float(read_values[10])))
            else:
                read_values[10] = -1

            for value in read_values[3:]:
                used_values += (float(value),)
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

            # replace time_stddev with it's log to make the value simpler
            if float(read_values[10]) >= 1.0:
                read_values[10] = str(math.log(float(read_values[10])))
            else:
                read_values[10] = -1

            for value in read_values[3:]:
                used_values += (float(value),)
            test_bidders_list.append(used_values)
            test_bidder_ids.append(read_values[0])


    # make a feature generator from sklearn to get all degree 3 features
    # poly = prep.PolynomialFeatures(2)

    # make arrays with the training data and normalize X
    train_X = np.array(train_bidders_list)
    # train_X = poly.fit_transform(train_X)
    # train_X = prep.normalize(train_X)
    train_y = np.array(outcomes_list)

    # train the thing
    # svr = svm.SVR(epsilon=0.04)
    dtr = DTR()
    dtr.fit(train_X[:-400],train_y[:-400])

    # make arrays with the test data and normalize X
    test_X = np.array(test_bidders_list)
    # test_X = poly.fit_transform(test_X)
    # test_X = prep.normalize(test_X)

    # predict for test data
    test_y = dtr.predict(test_X)


    # Kinda test the model
    errors = []
    fp = 0
    tp = 0
    fn = 0
    tn = 0
    cutoff = 0.5
    fuck_ups = []
    train_predict = dtr.predict(train_X)
    for h,y,bidder_id in zip(train_predict[-400:],train_y[-400:], train_bidder_ids[-400:]):
        print '%1.12f' % h, '--', y, '--', '%1.12f' % math.fabs(y-h)
        if math.fabs(y-h) > 0.5:
            fuck_ups.append((bidder_id, h, y))
        errors.append(math.fabs(y-h))
        if y > cutoff and h > cutoff:
            tp += 1
        elif y < cutoff and h < cutoff:
            tn += 1
        elif y < cutoff and h > cutoff:
            fp += 1
        else:
            fn += 1
    print '----------'
    print 'Average error:  ', sum(errors)/len(errors)
    print 'True positives: ', tp
    print 'True negatives: ', tn
    print 'False positives:', fp
    print 'False negatives:', fn
    print 'Cutoff:         ', cutoff
    print 'Fuck ups:'
    for fuck_up in fuck_ups:
        print ' ',fuck_up


    # print test_X
    # for y in test_y:
    #     print y

    # throw it in a csv file
    with open('../gen/submission.csv', 'w') as output_file:
        output_writer = csv.writer(output_file)
        titles = ('bidder_id','prediction')
        output_writer.writerow(titles)
        for prediction,bidder_id in zip(test_y,test_bidder_ids):
            output_writer.writerow([bidder_id, prediction])