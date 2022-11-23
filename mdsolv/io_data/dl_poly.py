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

    @staticmethod
    def read_HISTORY(filepath):
        history_data = {'steps':[],
                        'atoms':[],
                        'coordinates':[]}
        from monty.re import regrep
        with open(filepath) as f:
            data = f.readlines()
        history_datatype = int(data[1].strip().split()[0])
        patterns = {'timestep': r'timestep\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([-.\d]+)'}
        matches = regrep(str(filepath), patterns)
        history_data['timestep'] = float(matches['timestep'][0][0][4])
        strings_between_steps = matches['timestep'][1][1] - matches['timestep'][0][1]
        for i, match in enumerate(matches['timestep']):
            history_data['steps'].append(int(match[0][0]))
            nstr = int(match[1])
            if history_datatype == 0:
                j=0
                atoms_data = []
                coordinates_data = []
                while 4 + j*2 < strings_between_steps:
                    atoms_data.append(data[nstr + 4 + j*2].strip().split()[0])
                    coordinates_data.append(list(map(float, data[nstr + 5 + j*2].strip().split())))
                    j+=1
                history_data['atoms'].append(atoms_data)
                history_data['coordinates'].append(np.array(coordinates_data))
            else:
                raise ValueError('non 0 datatype of HISTORY file is not supported yet')
        return history_data
