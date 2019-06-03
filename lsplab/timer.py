from time import time


class timer(object):
    __start_time = None

    def __init__(self):
        self.__start_time = time()

    def elapsed(self):
        td = int(time() - self.__start_time)
        days = td // (3600 * 24)
        hours = (td // 3600) % 24
        minutes = (td // 60) % 60
        seconds = td % 60

        return '{0} days, {1} hours, {2} minutes, {3} seconds'.format(days, hours, minutes, seconds)
