#!/usr/bin/python

import os
import sys
import sqlite3
import datetime

import pandas as pd
import numpy as np

_PATH_TO_HERE = os.path.dirname(os.path.abspath(__file__))
PATH_TO_BCH_DB = os.path.join(_PATH_TO_HERE, "bchydro.db")

def load_remove_zeros(dbpath=PATH_TO_BCH_DB):
    """Read the load data from the given database. Return a Pandas time series
    containing the data, with zero-valued elements set to the last recorded
    non-zero value."""
    dataset = load(dbpath)
    return sg.data.remove_outlier_set_previous(dataset)

def load(dbpath=PATH_TO_BCH_DB):
    """Read the load data from the given database. Return a Pandas time series
    containing the data."""
    with sqlite3.connect(dbpath, detect_types=sqlite3.PARSE_DECLTYPES|
                             sqlite3.PARSE_COLNAMES) as conn:
        crs = conn.cursor()
        crs.execute("SELECT Timestamp as 'stamp [timestamp]', "\
                    "MWh as 'load [float]' "\
                    "FROM loads " \
                    "ORDER BY Timestamp ASC")
        stamps, loads = zip(*crs.fetchall())

    #import pdb; pdb.set_trace()
    return pd.Series(loads, index=stamps, dtype=float).asfreq('H')

def store_file(path):
    """Open a file, call LoadBCHydroStream to read and parse its contents, and
    finally store the data in the database."""
    with open(path, "r") as f:
        (timestamps, loads) = load_stream(f)
    storeindatabase(PATH_TO_BCH_DB, timestamps, loads)

def load_stream(stream):
    """Read a stream of lines containing dates, times and load values separated
    by semicolons. The date format is <alpha month> <numeric day>, <numeric
    year>. The time format is hour of day 1-24. Return a list of load values."""
    prev_timestamp = None
    loads = []
    timestamps = []
    for line in stream:
        (timestamp, load) = parseline(line)
        check_time_difference(prev_timestamp, timestamp)
        prev_timestamp = timestamp
        loads.append(load)
        timestamps.append(timestamp)
    return (timestamps, loads)

def check_time_difference(prev_timestamp, timestamp):
    """Check that the difference between previous and current timestamp is one
    hour."""
    if prev_timestamp is None:
        return
    if prev_timestamp + datetime.timedelta(hours=1) != timestamp:
        str_prev = prev_timestamp.strftime("%Y-%m-%d:%H")
        str_cur = timestamp.strftime("%Y-%m-%d:%H")
        raise RuntimeError("Time step error (0-based hours): '" +
                           str_prev + "' is not 1 hour before '"
                           + str_cur + "'.")
                           
def parseline(line, separator = ";"):
    """Expects a string with three (semicolon-)separated elements: a date
    (e.g. 'April 1, 2004', an hour of day (1-24), and a numeric value for that
    day. Returns a datetime and the value."""
    try:
        (datestr, hr, value) = line.split(separator)
    except ValueError as e:
        raise ValueError("Unable to split line into three items "
                         "(date, hr, value) using '" + separator +
                         "' as separator. Error message when "
                         "unpacking was: '" + e.args[0] + "'.")
    try:
        datetimestr = datestr + " " + str(int(hr)-1)
        d = datetime.datetime.strptime(datetimestr, "%B %d, %Y %H")
    except ValueError:
        raise ValueError("Unable to parse date and hour from string into "
                         "datetime. String was '" + datetimestr + ".")
    return (d, float(value))

def store_in_database(dbpath, timestamps, loads, stream=sys.stdout):
    """Creates a connection to the database used to store BC Hydro load
    data."""
    try:
        with sqlite3.connect(dbpath,
                             detect_types=sqlite3.PARSE_DECLTYPES) as conn:
            crs = conn.cursor()
            crs.execute("DELETE FROM loads")
            crs.executemany("INSERT INTO loads VALUES (?, ?)",
                            zip(timestamps, loads))
        print >>stream, "Stored %d fresh timestamps and loads " \
            "in database %s." % (len(loads), dbpath)
    except sqlite3.Warning as w:
        print >>sys.stderr, "Warning raised while trying to write loads and " \
            "timestamps to database. Message was: " + w.args[0]
    except sqlite3.Error as e:
        raise RuntimeError("Failed to write loads and timestamps to " \
                           "database. Error message was: " + e.args[0])


if __name__ == '__main__':
    from unittest import main
    main(module='test_'+__file__[:-3])
