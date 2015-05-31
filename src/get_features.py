__author__ = 'Marco Giancarli -- m.a.giancarli@gmail.com'


import numpy as np
import sqlite3 as sqlite
from decimal import Decimal
import math
import csv


TRAIN_TABLE_NAME = 'train'
TEST_TABLE_NAME = 'test'

BIDDERS_CONVERTERS = [str, str, str, float] # can be zipped with test or train
BIDS_CONVERTERS = [int, str, str, str, str, int, str, str, str]


def get_bidder_data(bidder_id, bidder_address, bidder_payment_account, cursor):
    num_bids = 0
    unique_auctions = {}
    unique_devices = {}
    unique_ips = {}
    unique_urls = {}
    unique_merchandise = {}
    unique_countries = {}
    times = []
    # num_same_address = 0
    # num_same_payment_account = 0
    time_since_last_bid_sum = 0
    bid_recency_sum = 0
    last_bid_count = 0

    # The code below is commented out because it seems not to be useful. It's
    # always 0 (with a bug that removes 1 for addresses in test data only)

    # if bidder_address not in get_bidder_data.address_dict:
    #     # count the number of people with the same address
    #     sql = 'SELECT COUNT(address) FROM %s WHERE address=?'
    #     num_same_address += int(
    #         cursor.execute(sql % TRAIN_TABLE_NAME, (bidder_address,)
    #                        ).fetchone()[0]
    #     )
    #     num_same_address += int(
    #         cursor.execute(sql % TEST_TABLE_NAME, (bidder_address,)
    #                        ).fetchone()[0]
    #     )
    #     num_same_address -= 1  # subtract 1 for the bidders themselves
    #     # save this data in a dict so that we don't have to count it again
    #     get_bidder_data.address_dict[bidder_address] = num_same_address
    # else:
    #     num_same_address = get_bidder_data.address_dict[bidder_address]
    #
    #
    # if bidder_payment_account not in get_bidder_data.payment_account_dict:
    #     # count the number of people with the same payment account
    #     sql = 'SELECT COUNT(payment_account) FROM %s WHERE payment_account=?'
    #     num_same_payment_account += int(
    #         cursor.execute(sql % TRAIN_TABLE_NAME, (bidder_payment_account,)
    #                        ).fetchone()[0]
    #     )
    #     num_same_payment_account += int(
    #         cursor.execute(sql % TEST_TABLE_NAME, (bidder_payment_account,)
    #                        ).fetchone()[0]
    #     )
    #     num_same_payment_account -= 1  # subtract 1 for the bidders themselves
    #     # save this data in a dict so that we don't have to count it again
    #     get_bidder_data.payment_account_dict[bidder_payment_account] = \
    #             num_same_payment_account
    # else:
    #     num_same_payment_account = \
    #             get_bidder_data.payment_account_dict[bidder_payment_account]


    sql = 'SELECT * FROM bids WHERE bidder_id=?'
    bids_from_this_bidder = cursor.execute(sql, (bidder_id,)).fetchall()

    for bid in bids_from_this_bidder:
        num_bids += 1

        auction =  bid[2]
        merchandise = bid[3]
        device = bid[4]
        time = int(bid[5])
        country = bid[6]
        ip = bid[7]
        url = bid[8]
        time_since_last_bid = bid[9]
        is_last_bid = bid[10]

        bid_data = [auction, merchandise, device, country, ip, url]
        unique_dicts = [unique_auctions, unique_merchandise, unique_devices,
                        unique_countries, unique_ips, unique_urls]

        for item, unique_dict in zip(bid_data, unique_dicts):
            if item not in unique_dict:
                unique_dict[item] = 1
            else:
                unique_dict[item] += 1

        times.append(time)
        time_since_last_bid_sum += time_since_last_bid
        bid_recency_sum += 1 / float(Decimal(time_since_last_bid+2).ln())
        last_bid_count += is_last_bid


    num_auctions = len(unique_auctions)
    num_devices = len(unique_devices)
    num_ips = len(unique_ips)
    num_urls = len(unique_urls)
    num_merchandise = len(unique_merchandise)
    num_countries = len(unique_countries)

    if len(times) > 1:
        # find the standard deviation of the intervals of time between bids.
        times.sort()
        time_diffs = [times[i+1] - times[i] for i in range(len(times)-1)]
        stddev = np.std(time_diffs)
        time_stddev = float(Decimal(stddev).ln()) if stddev > 2.72 else -100
    else:
        time_stddev = 0

    avg_time_since_last_bid = float(time_since_last_bid_sum) / \
                              float(num_bids) \
                              if num_bids > 0 else 0
    avg_bid_recency = float(bid_recency_sum) / \
                      float(num_bids) \
                      if num_bids > 0 else 0
    last_bid_rate = float(last_bid_count) / \
                    float(num_bids) \
                    if num_bids > 0 else 0

    return (
        math.log(num_bids + 1),
        math.log(num_auctions + 1),
        math.log(num_devices + 1),
        math.log(num_ips + 1),
        math.log(num_urls + 1),
        math.log(num_merchandise + 1),
        math.log(num_countries + 1),
        time_stddev,
        # num_same_address,
        # num_same_payment_account,
        avg_time_since_last_bid,
        avg_bid_recency,
        last_bid_rate
    )

# Use these to keep track of addresses and payment accounts that we already
# counted. Don't count them again.
get_bidder_data.address_dict = {}
get_bidder_data.payment_account_dict = {}


if __name__ == '__main__':
    connection = sqlite.connect('../res/auctions.db')
    cursor = connection.cursor()

    train_bidders = []
    test_bidders = []
    outcomes = []
    train_bidder_ids = []
    test_bidder_ids = []

    # load all of the data for the training set
    with open('../res/train.csv') as train:
        for line in train.readlines()[1:]:
            string_vals = line.split('\n')[0].split(',')
            vals = [f(val) for f,val in zip(BIDDERS_CONVERTERS,string_vals)]
            bidder = vals[:-1]
            outcome = vals[-1:]
            train_bidder_ids.append(bidder[0])
            train_bidders.append(tuple(bidder))
            outcomes.append(tuple(outcome))

    # load all of the data for the test set
    with open('../res/test.csv') as train:
        for line in train.readlines()[1:]:
            string_vals = line.split('\n')[0].split(',')
            bidder = [f(val) for f,val in zip(BIDDERS_CONVERTERS,string_vals)]
            test_bidder_ids.append(bidder[0])
            test_bidders.append(tuple(bidder))


    # bidder_index_map has bidder_ids for keys and the values are their respective
    # indices in the bidders lists. can be used for both the training set and the
    # test set because there shouldn't be any overlapping bidder_ids
    bidder_index_map = {}
    for bidder,index in zip(train_bidders, range(len(train_bidders))):
        bidder_index_map[bidder[0]] = index
    for bidder,index in zip(test_bidders, range(len(test_bidders))):
        bidder_index_map[bidder[0]] = index


    for bidder_id in train_bidder_ids:
        this_bidder = train_bidders[bidder_index_map[bidder_id]]
        this_bidders_address = this_bidder[2]
        this_bidders_payment_account = this_bidder[1]

        (
            num_bids,
            num_auctions,
            num_devices,
            num_ips,
            num_urls,
            num_merchandise,
            num_countries,
            time_stddev,
            # num_same_address,
            # num_same_payment_account,
            avg_time_since_last_bid,
            avg_bid_recency,
            last_bid_rate
        ) = get_bidder_data(bidder_id,
                            this_bidders_address,
                            this_bidders_payment_account,
                            cursor)

        train_bidders[bidder_index_map[bidder_id]] += (
            num_bids,
            num_auctions,
            num_devices,
            num_ips,
            num_urls,
            num_merchandise,
            num_countries,
            time_stddev,
            # num_same_address,
            # num_same_payment_account,
            avg_time_since_last_bid,
            avg_bid_recency,
            last_bid_rate,
            # float(num_bids) / float(num_auctions) if num_auctions > 0 else 0,
            # float(num_bids) / float(num_devices) if num_devices > 0 else 0,
            # float(num_bids) / float(num_ips) if num_ips > 0 else 0,
            # float(num_bids) / float(num_countries) if num_countries > 0 else 0,
            # float(num_bids) / float(num_urls) if num_urls > 0 else 0,
            # float(num_auctions) / float(num_devices) if num_devices > 0 else 0,
            # float(num_auctions) / float(num_ips) if num_ips > 0 else 0,
            # float(num_auctions) / float(num_countries) if num_countries > 0 else 0,
            # float(num_devices) / float(num_ips) if num_ips > 0 else 0,
            # float(num_auctions) / float(num_urls) if num_urls > 0 else 0,
        )


    for bidder_id in test_bidder_ids:
        this_bidder = test_bidders[bidder_index_map[bidder_id]]
        this_bidders_address = this_bidder[2]
        this_bidders_payment_account = this_bidder[1]

        (
            num_bids,
            num_auctions,
            num_devices,
            num_ips,
            num_urls,
            num_merchandise,
            num_countries,
            time_stddev,
            # num_same_address,
            # num_same_payment_account,
            avg_time_since_last_bid,
            avg_bid_recency,
            last_bid_rate
        ) = get_bidder_data(bidder_id,
                            this_bidders_address,
                            this_bidders_payment_account,
                            cursor)

        # add all the new data to this bidder's tuple
        test_bidders[bidder_index_map[bidder_id]] += (
            num_bids,
            num_auctions,
            num_devices,
            num_ips,
            num_urls,
            num_merchandise,
            num_countries,
            time_stddev,
            # num_same_address,
            # num_same_payment_account,
            avg_time_since_last_bid,
            avg_bid_recency,
            last_bid_rate,
            # float(num_bids) / float(num_auctions) if num_auctions > 0 else 0,
            # float(num_bids) / float(num_devices) if num_devices > 0 else 0,
            # float(num_bids) / float(num_ips) if num_ips > 0 else 0,
            # float(num_bids) / float(num_countries) if num_countries > 0 else 0,
            # float(num_bids) / float(num_urls) if num_urls > 0 else 0,
            # float(num_auctions) / float(num_devices) if num_devices > 0 else 0,
            # float(num_auctions) / float(num_ips) if num_ips > 0 else 0,
            # float(num_auctions) / float(num_countries) if num_countries > 0 else 0,
            # float(num_devices) / float(num_ips) if num_ips > 0 else 0,
            # float(num_auctions) / float(num_urls) if num_urls > 0 else 0,
        )


    # write all of the collected training data to a new csv file
    with open('../gen/train.csv', 'w') as output_file:
        output_writer = csv.writer(output_file)
        titles = ('bidder_id', 'payment_account', 'address', 'num_bids',
                  'num_auctions', 'num_devices', 'num_ips', 'num_urls',
                  'num_merchandise', 'num_countries', 'time_stddev',
                  # 'num_same_address', 'num_same_payment_account',
                  'avg_time_since_last_bid', 'avg_bid_recency', 'last_bid_rate',
                  # 'bids/auctions', 'bids/devices', 'bids/ips', 'bids/countries',
                  # 'bids/urls', 'auctions/devices', 'auctions/countries',
                  # 'auctions/countries', 'devices/ips', 'auctions/urls',
                  )
        output_writer.writerow(titles)
        for bidder in train_bidders:
            output_writer.writerow(bidder)

    # write all of the collected testing data to a new csv file
    with open('../gen/test.csv', 'w') as output_file:
        output_writer = csv.writer(output_file)
        titles = ('bidder_id', 'payment_account', 'address', 'num_bids',
                  'num_auctions', 'num_devices', 'num_ips', 'num_urls',
                  'num_merchandise', 'num_countries', 'time_stddev',
                  # 'num_same_address', 'num_same_payment_account',
                  'avg_time_since_last_bid', 'avg_bid_recency', 'last_bid_rate',
                  # 'bids/auctions', 'bids/devices', 'bids/ips', 'bids/countries',
                  # 'bids/urls', 'auctions/devices', 'auctions/countries',
                  # 'auctions/countries', 'devices/ips', 'auctions/urls',
                  )
        output_writer.writerow(titles)
        for bidder in test_bidders:
            output_writer.writerow(bidder)

    # write the outcomes to a separate file so that they are easy to load
    with open('../gen/outcomes.csv', 'w') as output_file:
        output_writer = csv.writer(output_file)
        titles = ('outcomes',)
        output_writer.writerow(titles)
        for outcome in outcomes:
            output_writer.writerow(outcome)