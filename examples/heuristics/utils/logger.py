import sys

class Tee(object):
    def __init__(self, name):
        self.file = open(name, "w")
        self.console = sys.stdout

    def write(self, data):
        self.console.write(data)  # Write to console
        self.file.write(data)     # Write to file

    def flush(self):
        self.console.flush()
        self.file.flush()

    def close(self):
        self.file.close()