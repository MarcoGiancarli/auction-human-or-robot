__author__ = 'Marco Giancarli -- m.a.giancarli@gmail.com'


import sqlite3 as sqlite


connection = sqlite.connect('../res/auctions.db')
cursor = connection.cursor()


print 'Dropping old tables and creating new ones in database...'

# create tables for the train, test, and bids data
cursor.execute('''DROP TABLE IF EXISTS train''')
cursor.execute('''CREATE TABLE train(bidder_id STRING PRIMARY KEY,
                                     payment_account STRING,
                                     address STRING,
                                     outcome REAL)''')

cursor.execute('''DROP TABLE IF EXISTS test''')
cursor.execute('''CREATE TABLE test(bidder_id STRING PRIMARY KEY ,
                                    payment_account STRING,
                                    address STRING)''')

cursor.execute('''DROP TABLE IF EXISTS bids''')
cursor.execute('''CREATE TABLE bids(bid_id INT PRIMARY KEY ,
                                    bidder_id STRING,
                                    auction STRING,
                                    merchandise STRING,
                                    device STRING,
                                    time INT,
                                    country STRING,
                                    ip STRING,
                                    url STRING,
                                    time_since_last_bid INT,
                                    is_last_bid INT)''')
cursor.execute('''CREATE INDEX bids_bidder_id ON bids (bidder_id)''')


connection.commit()
print 'Adding all tuples in CSV files to database...'

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
        values = [conv(val) for val,conv in zip(line.split(','), TRAIN_CONVERTERS)]
        # TODO: make this sql command safe
        sql = 'INSERT INTO train VALUES(' + ','.join(values) + ')'
        cursor.execute(sql)


# load bidders into test table
with open('../res/test.csv') as train:
    for line in train.readlines()[1:]:
        values = [conv(val) for val,conv in zip(line.split(','), TEST_CONVERTERS)]
        # TODO: make this sql command safe
        sql = 'INSERT INTO test VALUES(' + ','.join(values) + ')'
        cursor.execute(sql)


# load bids into bids table
with open('../res/bids.csv') as train:
    for line in train.readlines()[1:]:
        values = [conv(val) for val,conv in zip(line.split(','), BIDS_CONVERTERS)]
        # TODO: make this sql command safe
        sql = 'INSERT INTO bids VALUES(' + ','.join(values) + ',0,0' + ')'
        cursor.execute(sql)


connection.commit()
print 'Setting values for time_since_last_bid and is_last_bid...'

# create a new special cursor just for updating so that we can use the normal
# cursor as an iterator on the result
update_cursor = connection.cursor()

# we must get the time_since_last_bid and is_last_bid values for the bids table
# by querying the entire bids table after it has been populated
sql = 'SELECT bid_id,auction,time FROM bids ORDER BY auction,time'
cursor.execute(sql)

prev_bid = (-1, '', -1)
for bid in cursor.execute(sql):
    # set is_last_bid to 1 by default and fix it if it isn't later
    sql = 'UPDATE bids SET is_last_bid=1 WHERE bid_id=?'
    update_cursor.execute(sql, (bid[0],))

    # if this is a new auction, this is the first bid in the auction
    if bid[1] != prev_bid[1]:
        # set time_since_last_bid to be -1
        sql = 'UPDATE bids SET time_since_last_bid=0 WHERE bid_id=?'
        update_cursor.execute(sql, (bid[0],))
    else:
        # set time_since_last_bid to be the time difference
        time_diff = long(bid[2]) - long(prev_bid[2])
        sql = 'UPDATE bids SET time_since_last_bid=? WHERE bid_id=?'
        update_cursor.execute(sql, (time_diff, bid[0]))
        # set is_last_bid of prev_bid to be 1
        sql = 'UPDATE bids SET is_last_bid=0 WHERE bid_id=?'
        update_cursor.execute(sql, (prev_bid[0],))

    prev_bid = bid


connection.commit()
cursor.close()
connection.close()

print 'Finished creating database.'