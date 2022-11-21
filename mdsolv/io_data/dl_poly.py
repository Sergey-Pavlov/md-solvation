import numpy as np
from mdsolv.core.structure import Trajectory

class DL_POLY():

    def __init__(self, system):
        self.system = system

    def write_CONFIG(self, filename):
        with open(filename, 'w') as f:
            f.write('DL_POLY CONFIG File' + '\n' +
                          '         0         3\n' +
                          "%20f%20f%20f" % tuple(self.system.box[0]) + '\n' +
                          "%20f%20f%20f" % tuple(self.system.box[1]) + '\n' +
                          "%20f%20f%20f" % tuple(self.system.box[2]) + '\n')
            for i in range(len(self.system.atomlist)):
                f.write("%3s%15s%10s" % (self.system.atomlist[i], i+1, 0) + '\n')
                f.write("%20f%20f%20f" % tuple(self.system.coordinates[i]) + '\n')
