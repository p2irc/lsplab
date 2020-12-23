import emoji


class reporter(object):
    __steps = []
    __success_emoji = ':heavy_check_mark:'
    __failure_emoji = ':heavy_multiplication_x:'

    def __init__(self):
        self.__steps = []

    def add(self, message, success):
        self.__steps.append((message, success))

    def print_all(self):
        for item in self.__steps:
            status_mark = self.__success_emoji if item[1] else self.__failure_emoji
            print(emoji.emojize("{0} {1}".format(status_mark, item[0])))

    def all_succeeded(self):
        return all([item[1] for item in self.__steps])