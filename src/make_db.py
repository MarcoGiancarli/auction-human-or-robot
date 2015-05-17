__author__ = 'Marco Giancarli -- m.a.giancarli@gmail.com'


import sqlite3 as sqlite


connection = sqlite.connect('../res/auctions.db')
cursor = connection.cursor()


# create tables for the train, test, and bids data
cursor.execute('''DROP TABLE IF EXISTS train;''')
cursor.execute('''CREATE TABLE train(bidder_id STRING PRIMARY KEY,
                                     payment_account STRING,
                                     address STRING,
                                     outcome REAL);''')

cursor.execute('''DROP TABLE IF EXISTS test;''')
cursor.execute('''CREATE TABLE test(bidder_id STRING PRIMARY KEY ,
                                    payment_account STRING,
                                    address STRING);''')

cursor.execute('''DROP TABLE IF EXISTS bids;''')
cursor.execute('''CREATE TABLE bids(bid_id INT PRIMARY KEY ,
                                    bidder_id STRING,
                                    auction STRING,
                                    merchandise STRING,
                                    device STRING,
                                    time INT,
                                    country STRING,
                                    ip STRING,
                                    url STRING);''')
cursor.execute('''CREATE INDEX bids_bidder_id ON bids (bidder_id);''')


# make converter functions for each value in each tuple
def do_nothing(x):
    return x

def add_quotes(x):
    return "'" + x + "'"

TRAIN_CONVERTERS = [add_quotes, add_quotes, add_quotes, do_nothing]
TEST_CONVERTERS = [add_quotes, add_quotes, add_quotes]
BIDS_CONVERTERS = [do_nothing, add_quotes, add_quotes, add_quotes, add_quotes,
                   do_nothing, add_quotes, add_quotes, add_quotes]

# load bidders into train table
with open('../res/train.csv') as train:
    for line in train.readlines()[1:]:
        values = ','.join(
            [conv(val) for val,conv in zip(line.split(','), TRAIN_CONVERTERS)]
        )
        # TODO: make this sql command safe
        sql = 'INSERT INTO train VALUES(' + values + ')'
        cursor.execute(sql)


# load bidders into test table
with open('../res/test.csv') as train:
    for line in train.readlines()[1:]:
        values = ','.join(
            [conv(val) for val,conv in zip(line.split(','), TEST_CONVERTERS)]
        )
        # TODO: make this sql command safe
        sql = 'INSERT INTO test VALUES(' + values + ')'
        cursor.execute(sql)


# load bids into bids table
with open('../res/bids.csv') as train:
    for line in train.readlines()[1:]:
        values = ','.join(
            [conv(val) for val,conv in zip(line.split(','), BIDS_CONVERTERS)]
        )
        # TODO: make this sql command safe
        sql = 'INSERT INTO bids VALUES(' + values + ')'
        cursor.execute(sql)


connection.commit()
cursor.close()
connection.close()