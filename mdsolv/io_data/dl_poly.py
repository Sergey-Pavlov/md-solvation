import numpy as np
from mdsolv.core.structure import Trajectory

class DL_POLY():

    def __init__(self, system, coordinates):
        self.system = system

    def write_CONFIG(self, filename):
        with open(filename, 'w') as f:
            f.write('DL_POLY CONFIG File' + '\n' +
                          '         0         3\n' +
                          "%20f%20f%20f" % (self.system.box[0]) + '\n' +
                          "%20f%20f%20f" % (self.system.box[1]) + '\n' +
                          "%20f%20f%20f" % (self.system.box[2]) + '\n')
