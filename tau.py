import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pytrr
import json
import glob
from functools import lru_cache
from scipy.ndimage.filters import uniform_filter1d
from scipy.optimize import curve_fit

def mass_define(address):
    """
    .top parses
    get mass data from .top file
    input:
        address: path to .top file
    output:
        dict with mass
    """
    with open(address, 'r') as top:
        dict_mass = {}
        for line in top:
            if line.strip() == '[ atoms ]':
                for line in top:
                    if line.strip().startswith(';'):
                        continue
                    elif line.isspace():
                        break
                    dict_mass.setdefault(line.split()[4].strip(), float(line.split()[7].strip()))
    return dict_mass

def system_define(address):
    """
    .gro parser
    read gro file and get dict molecules-atoms
    return:
        system - dict "key" - molecule, item - list of atoms
    """
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

def system_define_v2(address):
    df = pd.read_fwf(address, widths=[9,6,5,8,8,8,8,8,8], header=None, index_col=(0,1), skiprows=2, skipfooter=1)
    system={}
    for multiindex in df.index.values:
        try:
            system[multiindex[0]].append(multiindex[1])
        except KeyError:
            system[multiindex[0]]=[multiindex[1]]
    return system

def neighbors(address, system, mass, ion, sel, rdf):
    """
    Open traj.trr
    find centers of mass of molecules
    neighbors - [step, list of ions, list of neighbors]
    """
    ions = list(filter(lambda s: ion in s, system))
    with pytrr.GroTrrReader(trr_file) as trajectory:
        neighbors = []
        for step, frame in enumerate(trajectory):
            neighbors.append([])
            coordinates = {}
            done = 0
            frame_data = trajectory.get_data()
            box = [frame_data['box'][i][i] for i in range(0,3)]
            #df1 = pd.concat([system_stacked, pd.DataFrame(frame_data['x'], index = system_stacked.index)], axis=1)
            for molecule in system:
                atoms = [[frame_data['x'][i+done][0],
                          frame_data['x'][i+done][1],
                          frame_data['x'][i+done][2],
                          mass[atom]] for i, atom in enumerate(system[molecule])]
                coordinates[molecule] = [sum(map(lambda a: a[i] * a[3], atoms)) /
                                         sum(map(lambda a: a[3], atoms)) for i in range(0,3)]
                done += len(system[molecule])
            df = pd.DataFrame.from_dict(coordinates, orient='index')
            df1 = df.filter(regex=sel, axis=0)
            names = df1.index.values
            for i, n in enumerate(ions):
                coord = df.loc[n].values
                x=df1.to_numpy()
                x=x-coord
                x=np.abs(x)
                x=np.where(x+x < box, x, box-x)
                nt = np.linalg.norm(x, axis = 1) < rdf
                neighbors[step].append(list(names[nt]))
                #try:
                #    neighbors[step][i].remove(n)
                #except ValueError:
                #    pass
    return neighbors

def result_assembler(neighbors):
    """
    return:
    array for each ions list of number of neigbors
    """
    # TODO: check this function
    result = []
    num_of_frames = len(neighbors)
    num_of_ref = len(neighbors[0])
    for start_frame in range(0, num_of_frames // 3 * 2, num_of_frames // 10):
        common_neighbors = [neighbors[start_frame]]
        for common, whole, step in zip(common_neighbors, neighbors[start_frame+1 :], range(num_of_frames // 10)):
            common_neighbors.append([list(set(common[i]) & set(whole[i])) for i in range(num_of_ref)])
        result.extend([[len(step[i]) for step in common_neighbors] for i in range(num_of_ref)])
    return result

def write_neighbors(address, neighbors):
    with open(jsonfile, 'w') as data_file:
        json.dump(neighbors, data_file)

def read_neighbors(address):    
    with open(jsonfile, 'r') as read_file:
        return json.load(read_file)

def plot_exp(result):
    result2 = np.array(result)
    exp = np.mean(result2, axis=0)
    return exp

def func(x, a, t):
    return a*np.exp(-x/t)

def fu(indf):
    df = uniform_filter1d(indf, size=20, mode='reflect')
    return pd.Series(df, index = indf.index[0:len(df)])

@lru_cache(maxsize=2)
def names(address):    
    with open(address, 'r') as f:
        labels = []
        for i, line in enumerate(f):
            if line.startswith('#'):
                continue
            if line.startswith('@'):
                if (line[2] == 's') and (line.split().count('legend')):
                    labels.append(line.split()[line.split().index('legend') + 1].strip('"'))
                continue
            break
        return i, labels

label = ['0', '20', '40', '60', '80']
path = 'C:\\Users\\Slava\\Documents\\k\\DMSO\\NO3\\RDF\\'
mask = 'Li*.xvg'
files = glob.glob(path+mask)
#print(files)
rdf_list = [pd.read_fwf(file, skiprows=[i for i in range(0, names(file)[0])], index_col=0, header=None, names = names(file)[1]) for file in files]
rdf_Li = dict.fromkeys(names(files[1])[1])
rdf_Li['DMSO'] = pd.concat([rdf_list[i].loc[:,'DMSO'].rename(label[i]) for i in range(0, len(label))], axis = 1)
rdf_Li['NO3'] = pd.concat([rdf_list[i].loc[:,'NO3-'].rename(label[i]) for i in range(0, len(label)-1)], axis = 1)
rdf_Li['TFSI'] = pd.concat([rdf_list[i].loc[:,'TFS'].rename(label[i]) for i in range(1, len(label))], axis = 1)
mask = 'O2*.xvg'
files = glob.glob(path+mask)
#print(files)
rdf_list = [pd.read_fwf(file, skiprows=[i for i in range(0, names(file)[0])], index_col=0, header=None, names = names(file)[1]) for file in files]
rdf_O2 = dict.fromkeys(names(files[1])[1])
rdf_O2['DMSO'] = pd.concat([rdf_list[i].loc[:,'DMSO'].rename(label[i]) for i in range(0, len(label))], axis = 1)
rdf_O2['NO3'] = pd.concat([rdf_list[i].loc[:,'NO3-'].rename(label[i]) for i in range(0, len(label)-1)], axis = 1)
rdf_O2['TFSI'] = pd.concat([rdf_list[i].loc[:,'TFS'].rename(label[i]) for i in range(1, len(label))], axis = 1)
with pd.ExcelWriter(r'C:\Users\Slava\Documents\k\DMSO\NO3\RDF\RDF.xls') as writer:
    rdf_Li['DMSO'].to_excel(writer, sheet_name="Li_DMSO")
    rdf_Li['NO3'].to_excel(writer, sheet_name="Li_NO3")
    rdf_Li['TFSI'].to_excel(writer, sheet_name="Li_TFSI")
    rdf_O2['DMSO'].to_excel(writer, sheet_name="O2_DMSO")
    rdf_O2['NO3'].to_excel(writer, sheet_name="O2_NO3")
    rdf_O2['TFSI'].to_excel(writer, sheet_name="O2_TFSI")

a = 'DMSO'
b = '20'
rdf_O2[a].apply(fu).loc[rdf_O2[a].apply(fu)[b] == rdf_O2[a].apply(fu).loc[0.5:0.8,b].min()].index.values[0]

folder = 'C:\\Users\\Slava\\Documents\\k\\DMSO\\NO3\\20_TFSI\\'
top_file = folder + 'DMSO_TFSI_Li+_O2-.top'
gro_file = folder + 'DMSO_20TFSI_3.gro'
trr_file = folder + 'DMSO_20TFSI_30.trr'
jsonfile = folder + 'DMSO_20TFSI_O2_neighbors.json'
ion = 'O2'
sel = 'DMSO'
rdf = 0.622

t1 = time.perf_counter()
mass = mass_define(top_file)
system = system_define(gro_file)
#df_system = pd.DataFrame.from_dict(system, orient='index')
#system_stacked = df_system.stack()
tau1 = neighbors(trr_file, system, mass, ion, sel, rdf)
#write_neighbors(json_file, tau1)
t2 = time.perf_counter()
print('{:.0f}:{:05.2f}'.format((t2-t1)//60,(t2-t1)%60))

ts = 0.01 # ns
result = result_assembler(read_neighbors(jsonfile))
#print(result[:5])
x_data = np.arange(0.0, len(result[0])*ts, ts)
popt, pcov = curve_fit(func, x_data, plot_exp(result), [1, 1])
print(b,a, *popt)
#plt.figure(figsize=(10, 8))
plt.plot(x_data, plot_exp(result), x_data, func(x_data, *popt), label='Fitted function')
plt.yscale('log')
plt.legend(loc='best')
plt.show()

rdf_O2['DMSO'].plot()
#plt.axis([0.5, 0.8, 0.5, 1])
to_plot = rdf_O2['DMSO'].apply(fu)
#plt.figure(figsize=(10, 8))
to_plot.plot()
#plt.axis([0.5, 0.8, 0.5, 1])