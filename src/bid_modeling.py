__author__ = 'Marco Giancarli -- m.a.giancarli@gmail.com'


from sklearn.svm import SVR
from sklearn import preprocessing as prep
from sklearn.metrics import roc_auc_score

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import sqlite3 as sqlite
import numpy as np
import csv


if __name__=='__main__':
    connection = sqlite.connect('../res/auctions.db')
    cursor = connection.cursor()

    robot_bids = []


    sql = 'SELECT bidder_id FROM train WHERE outcome=1.0'
    robot_bidder_ids = [result[0] for result in cursor.execute(sql)]
    print len(robot_bidder_ids),'robots found.'

    for robot_bidder_id in robot_bidder_ids:
        sql = 'SELECT time_since_last_bid,time FROM bids WHERE bidder_id=?'
        bids = cursor.execute(sql, (robot_bidder_id,)).fetchall()
        robot_bids.extend(bids)
    robot_bids = [(float(val[0]), float(val[1])) for val in robot_bids]

    print len(robot_bids), 'robot bids found.'

    sql = 'SELECT time_since_last_bid,time FROM bids ' + \
          'INNER JOIN train ON bids.bidder_id=train.bidder_id ' + \
          'WHERE outcome=0.0 ' + \
          'LIMIT 600000'
    human_bids = cursor.execute(sql).fetchall()
    human_bids = [(float(val[0]), float(val[1])) for val in human_bids]

    print len(human_bids), 'human bids collected.'


    # stack the X values
    robot_X = np.array(robot_bids)
    human_X = np.array(human_bids)
    X = np.vstack((robot_X, human_X))

    # stack the y values
    robot_y = np.ones((len(robot_bids),))
    human_y = np.zeros((len(human_bids),))
    y = np.hstack((robot_y, human_y))

    # shuffle the arrays to the same permutation
    rng_state = np.random.get_state()
    np.random.shuffle(X)
    np.random.set_state(rng_state)
    np.random.shuffle(y)

    # scale the data
    # X = prep.scale(X)

    print X[:20]

    colors = {0.0: 'g', 1.0: 'r'}
    plt.scatter(X[:10000,0], X[:10000,1], c=[colors[outcome] for outcome in y], alpha=0.3)
    plt.show()

    model = SVR()
    divider = 5000
    model.fit(X[:divider], y[:divider])

    predictions = model.predict(X[-divider:])
    print roc_auc_score(y[-divider:], predictions)