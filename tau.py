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
        dict with atom mass
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

def box_f(address):
    with pytrr.GroTrrReader(address) as trajectory:
        for step, frame in enumerate(trajectory):
            frame_data = trajectory.get_data()
            return frame_data['box'].diagonal()

def mol_com(address, system, mass):
    """
    .trr parser
    read trr file and get molecules centers of mass coordinates
    input:
        address: path to .trr file
        system: dict "key" - molecule, item - list of atoms
        mass: dict with atom mass
    output:
        array of molecules com coordinates
        (num_of_steps, num_of_molecues, 3[x,y,z]) shaped
    """
    com = []
    mass_arr = [np.array([[mass[atom]]*3 for atom in molecule]) for molecule in system.values()]#массив масс атомов в форме массива координат атомов сгрупированных по молекулам
    mass_mod = [mol_mass/np.sum(mol_mass, axis=0) for mol_mass in mass_arr]#приведённый на массу молекул массив масс атомов
    mass_1=np.array([j for i in mass_mod for j in i])
    chunks = np.array([len(molecule) for molecule in system.values()])
    stop = np.cumsum(chunks)#границы молекул в массиве координат атомов
    start = stop[:-1]#границы молекул в массиве координат атомов
    with pytrr.GroTrrReader(address) as trajectory:
        for frame in trajectory:
            frame_data = trajectory.get_data()
            box = frame_data['box'].diagonal()
            box2=box/2
            mask = [(np.ptp(atoms, axis=0) > box2) & (atoms < box2) for atoms in np.vsplit(frame_data['x'],start)]
            mask = np.array([j for i in mask for j in i])
            atoms = np.where(mask, frame_data['x']+box, frame_data['x'])
            atoms = np.multiply(atoms,mass_1)
            x = np.array([np.sum(i, axis=0) for i in np.vsplit(atoms,start)])
            com.append(np.where(x < box, x, x-box))
    return np.array(com)

def neighbors(com, box, system, ref, sel, rdf):
    """
    get sel-type entities within rdf radius of ref-type entities,
    presented in system, com - coordinates for system
    neighbors - [step, list of ions, list of neighbors]
    """    
    mask_ref = np.array([(ref in i) for i in system.keys()])
    mask_sel = np.array([(sel in i) for i in system.keys()])
    if ~np.any(mask_ref):
        raise KeyError('No {} in system'.format(ref))
    if ~np.any(mask_sel):
        raise KeyError('No {} in system'.format(sel))
    names = np.array(list(system.keys()))[mask_sel]
    neighbors = []
    for step, coordinates in enumerate(com):
        neighbors.append([])
        coord=coordinates[mask_sel]
        for n in coordinates[mask_ref]:
            x=np.abs(coord-n)
            x=np.where(x+x < box, x, box-x)
            nt = np.linalg.norm(x, axis = 1) < rdf
            neighbors[step].append(set(names[nt]))
    return neighbors

def result_assembler(neighbors, lf, start_step, limit, num_of_frames, num_of_ref):
    """
    lf - tolerable escape time in timesteps
    start_step - timesteps between restarts
    limit - lenth of investigated trajectory in timesteps
    return:
    tau, tau_std, coord_num, coord_num_srd
    """
    # TODO: check this function
    if num_of_frames < limit+lf:
        raise ValueError('limit is more than trj')
    result = []
    coord = []
    for start_frame in range(0, num_of_frames - (limit+lf+1), start_step):
        common_neighbors = [neighbors[start_frame]]
        for num, common in zip(range(limit), common_neighbors):
            view = list(map(set.union,*neighbors[start_frame+1+num:start_frame+1+num+lf]))
            common_neighbors.append([common[ref_i].intersection(view[ref_i]) for ref_i in range(num_of_ref)])
        coord.extend([len(common_neighbors[0][ref_i]) for ref_i in range(num_of_ref)])
        result.extend([[len(step[ref_i]) for step in common_neighbors] for ref_i in range(num_of_ref)])
    coord_np_arr = np.array(coord)
    result_np_arr = np.array(result)
    return np.mean(result_np_arr, axis=0), np.std(result_np_arr, axis=0), np.mean(coord_np_arr), np.std(coord_np_arr)

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