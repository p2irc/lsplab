from time import time
from datetime import timedelta

class timer(object):
    __start_time = None

    def __init__(self):
        self.__start_time = time()

    def elapsed(self):
        td = timedelta(seconds=(time() - self.__start_time))
        days = td.seconds // (3600 * 24)
        hours = td.seconds // 3600
        minutes = (td.seconds // 60) % 60
        seconds = td.seconds % 60

        return '{0} days, {1} hours, {2} minutes, {3} seconds'.format(days, hours, minutes, seconds)
