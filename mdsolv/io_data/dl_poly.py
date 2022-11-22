import numpy as np
from mdsolv.core.structure import Trajectory

class DL_POLY():

    def __init__(self, system):
        self.system = system

    def write_CONFIG(self, filename):
        with open(filename, 'w') as f:
            f.write('CFGEDT CONFIG File' + '\n' +
                    f'         0         2      {len(self.system.atomlist)}    0' + '\n' +
                    "%20.12f%20.12f%20.12f" % tuple(self.system.box[0]) + '\n' +
                    "%20.12f%20.12f%20.12f" % tuple(self.system.box[1]) + '\n' +
                    "%20.12f%20.12f%20.12f" % tuple(self.system.box[2]) + '\n')
            for i in range(len(self.system.atomlist)):
                f.write("%3s%15s" % (self.system.atomlist[i], i+1) + '\n')
                f.write("%20.9f%20.9f%20.9f" % tuple(self.system.coordinates[i]) + '\n')
