import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytrr
import MDAnalysis
import json

t1 = time.perf_counter()
top_file = r'C:\Users\Slava\Documents\k\DMSO\NO3\40_TFSI\DMSO_TFSI_Li+_O2-.top'
gro_file = r'C:\Users\Slava\Documents\k\DMSO\NO3\40_TFSI\DMSO_40TFSI_2.gro'
tpr_file = r'C:\Users\Slava\Documents\k\DMSO\NO3\40_TFSI\DMSO_40TFSI_2.tpr'
trr_file = r'C:\Users\Slava\Documents\k\DMSO\NO3\40_TFSI\DMSO_40TFSI_1.trr'
json_file = r'C:\Users\Slava\Documents\k\DMSO\NO3\40_TFSI\DMSO_40TFSI_Li_neighbors.json'
ion = 'Li'
rdf = 0.41

def mass_define(address):
    with open(address, 'r') as top:
        dict_mass = {}
        for line in top:
            if line == '[ atoms ]\n':
                for line in top:
                    if ';' in line:
                        continue
                    elif line.isspace():
                        break
                    dict_mass.setdefault(line.split()[4].strip(), float(line.split()[7].strip()))
    return dict_mass

def system_define(address):
    with open(address, 'r') as gro:
        system = {} # Starting from Python 3.7, insertion order of Python dictionaries is guaranteed.
        gro.readline()
        gro.readline()
        for line in gro:
            try:
                system[line[0:9].strip()].append(line[10:15].lstrip())
            except KeyError:
                system.setdefault(line[0:9].strip(), [line[10:15].lstrip()])
        system.popitem() # The popitem() method removes the item that was last inserted into the dictionary.
                        # In versions before 3.7, the popitem() method removes a random item.
    return system

def in_distance(n, m, box, rdf):
    def dist(x, l): return x if x + x < l else l - x
    if dist(abs(n[0] - m[0]), box[0]) > rdf: return False
    elif dist(abs(n[1] - m[1]), box[1]) > rdf: return False
    elif dist(abs(n[2] - m[2]), box[2]) > rdf: return False
    elif sum(map(lambda x1, x2, l: dist(abs(x1 - x2), l)**2, n, m, box)) > rdf * rdf: return False
    else: return True

def neighbors(address, system, mass, ion, rdf):
    with pytrr.GroTrrReader(address) as trajectory:
        ions = list(filter(lambda s: ion in s, system))
        neighbors = []
        for step, frame in enumerate(trajectory):
            neighbors.append([])
            coordinates = {}
            prev = 0
            frame_data = trajectory.get_data()
            box = [frame_data['box'][i][i] for i in range(0,3)]
            for molecule in system:
                atoms = [[frame_data['x'][i+prev][0],
                          frame_data['x'][i+prev][1],
                          frame_data['x'][i+prev][2],
                          mass[atom]] for i, atom in enumerate(system[molecule])]
                coordinates[molecule] = [sum(map(lambda a: a[i] * a[3], atoms)) /
                                         sum(map(lambda a: a[3], atoms)) for i in range(0,3)]
                prev += len(system[molecule])
            for i, n in enumerate(ions):
                neighbors[step].append([])
                coord = coordinates[n]
                for k in filter(lambda m: in_distance(coord, coordinates[m], box, rdf), coordinates): neighbors[step][i].append(k)
                try:
                    neighbors[step][i].remove(n)
                except ValueError:
                    pass
    return neighbors

def result_assembler(neighbors):
    result = []
    molnum = 0
    frames = len(neighbors)
    n = len(neighbors[0])
    for start_frame in range(0, frames // 3 * 2, frames // 10):
        tau2 = [neighbors[start_frame]]
        result.extend([[] for i in range(n)])
        for step in range(start_frame, start_frame + frames // 3):
            tau2.append([list(set(tau2[step - start_frame][i]) & set(neighbors[step][i])) for i in range(n)])
            for i in range(n):
                result[molnum + i].append(len(tau2[step - start_frame + 1][i]))
        molnum += n
    return result

def write_neighbors(address, neighbors):
    with open(json_file, 'w') as data_file:
        json.dump(neighbors, data_file)

def read_neighbors(address):    
    with open(json_file, 'r') as read_file:
        return json.load(read_file)

def plot_exp(result):
    result2 = np.array(result)
#    print(result2)
    exp = np.mean(result2, axis=0)
    return exp

mass = mass_define(top_file)
system = system_define(gro_file)
tau1 = neighbors(trr_file, system, mass, ion, rdf)
#write_neighbors(json_file, tau1)
print(len(tau1))
t2 = time.perf_counter()
print('{:.0f}:{:05.2f}'.format((t2-t1)//60,(t2-t1)%60))
