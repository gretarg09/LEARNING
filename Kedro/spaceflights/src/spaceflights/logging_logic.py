import logging
import rich


class MyCustomFormatter(logging.Formatter):
    '''My beautiful custom formatter'''
    

    def format(self, record):

        rich.inspect(record)

        record.asctime = self.formatTime(record, self.datefmt)
        custom_message = f"Custom logger -- {record.asctime} - {record.name} - {record.levelname} - {record.msg}"

        return custom_message
