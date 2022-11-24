import numpy as np
import pandas as pd
import pytrr
import itertools

class Interface_Analyser:

    def __init__(self):
        pass

    def mass_define(self, filepath):
        with open(filepath, 'r') as top:
            dict_mass = {}
            for line in top:
                if line.strip() == '[ atoms ]':
                    for line in top:
                        if line.strip().startswith(';'):
                            continue
                        elif line.isspace():
                            break
                        dict_mass.setdefault(line.split()[4].strip(), float(line.split()[7].strip()))
        self.masses = dict_mass

    def charges_define(self, filepath):
        with open(filepath, 'r') as top:
            dict_charges = {}
            for line in top:
                if line.strip() == '[ atoms ]':
                    for line in top:
                        if line.strip().startswith(';'):
                            continue
                        elif line.isspace():
                            break
                        dict_charges.setdefault(line.split()[3].strip()[0]+line.split()[4].strip(), float(line.split()[6].strip()))
        self.charges = dict_charges

    def system_define(self, filepath):
        df = pd.read_fwf(filepath, widths=[9,6,5,8,8,8,8,8,8], header=None, index_col=(0,1), skiprows=2, skipfooter=1)
        system={}
        for multiindex in df.index.values:
            try:
                system[multiindex[0]].append(multiindex[1])
            except KeyError:
                system[multiindex[0]]=[multiindex[1]]
        self.system = system

    def calculate_local_density(self, traj_path, dx):
        traj_data = []
        system_flatten = list(itertools.chain(*list(self.system.values())))
        masses_flatten = np.array([self.masses[atom] for atom in system_flatten])
        with pytrr.GroTrrReader(traj_path) as trajectory:
            for step, frame in enumerate(trajectory):
                coordinates = []
                frame_data = trajectory.get_data()
                if step == 0:
                    box = np.array([frame_data['box'][i][i] for i in range(0, 3)])
                traj_data.append(frame_data['x'])
        traj_data = np.array(traj_data)
        bins = (np.array([6.273, 4.26]) // dx).astype(int)
        steps = traj_data.shape[0]
        traj_data = traj_data.reshape(-1, 3)
        hist, x_edges, y_edges = np.histogram2d(traj_data[:, 2], traj_data[:, 1], bins=bins,
                                                range=[[0.001, 6.272], [0.0, 4.26]],
                                                weights=np.tile(masses_flatten, steps))
        hist_z, bins_z = np.histogram(traj_data[:, 2], bins=bins[0], range=[0.001, 6.272],
                                      weights=np.tile(masses_flatten, steps))
        volume = box[0] * (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        volume_z = box[0] * box[1] * (bins_z[1] - bins_z[0])
        Da = 1.66053906660e-24  # Da in gramms
        nm_to_cm = 1e-7
        hist = hist * Da / volume / steps / nm_to_cm ** 3
        hist_z = hist_z * Da / volume_z / steps / nm_to_cm ** 3
        return hist, x_edges, y_edges, hist_z, bins_z

    def calculate_charge_density(self, traj_path, dx):
        traj_data = []
        mol_name_letters = []
        for s in list(self.system.keys()):
            i = s.find(next(filter(str.isalpha, s)))
            mol_name_letters.append(s[i])
        system_flatten = []
        for i, name in enumerate(list(self.system.values())):
            for atom in name:
                system_flatten.append(mol_name_letters[i] + atom)
        charges_flatten = np.array([self.charges[atom] for atom in system_flatten])
        with pytrr.GroTrrReader(traj_path) as trajectory:
            for step, frame in enumerate(trajectory):
                coordinates = []
                frame_data = trajectory.get_data()
                if step == 0:
                    box = np.array([frame_data['box'][i][i] for i in range(0, 3)])
                traj_data.append(frame_data['x'])
        traj_data = np.array(traj_data)
        bins = (np.array([6.273, 4.26]) // dx).astype(int)
        steps = traj_data.shape[0]
        traj_data = traj_data.reshape(-1, 3)
        hist, x_edges, y_edges = np.histogram2d(traj_data[:, 2], traj_data[:, 1], bins=bins,
                                                range=[[0.001, 6.272], [0.0, 4.26]],
                                                weights=np.tile(charges_flatten, steps))
        hist_z, bins_z = np.histogram(traj_data[:, 2], bins=bins[0], range=[0.001, 6.272],
                                      weights=np.tile(charges_flatten, steps))
        volume = box[0] * (x_edges[1] - x_edges[0]) * (y_edges[1] - y_edges[0])
        volume_z = box[0] * box[1] * (bins_z[1] - bins_z[0])
        Da = 1.66053906660e-24  # Da in gramms
        nm_to_cm = 1e-7
        hist = hist * Da / volume / steps / nm_to_cm ** 3
        hist_z = hist_z * Da / volume_z / steps / nm_to_cm ** 3
        return hist, x_edges, y_edges, hist_z, bins_z

    def calculate_local_density_1d(self, traj_path, dx):
        traj_data = []
        system_flatten = list(itertools.chain(*list(self.system.values())))
        masses_flatten = np.array([self.masses[atom] for atom in system_flatten])
        with pytrr.GroTrrReader(traj_path) as trajectory:
            for step, frame in enumerate(trajectory):
                coordinates = []
                frame_data = trajectory.get_data()
                if step == 0:
                    box = np.array([frame_data['box'][i][i] for i in range(0, 3)])
                traj_data.append(frame_data['x'])
        traj_data = np.array(traj_data)
        bins = (np.array([6.273, 4.26]) // dx).astype(int)
        steps = traj_data.shape[0]
        traj_data = traj_data.reshape(-1, 3)
        hist_z, bins_z = np.histogram(traj_data[:, 2], bins=bins[0], range=[0.001, 6.272],
                                      weights=np.tile(masses_flatten, steps))
        volume_z = box[0] * box[1] * (bins_z[1] - bins_z[0])
        Da = 1.66053906660e-24  # Da in gramms
        nm_to_cm = 1e-7
        hist_z = hist_z * Da / volume_z / steps / nm_to_cm ** 3
        return hist_z, bins_z

    def calculate_local_density_1d_mask(self, traj_path, dx, mol_type=None, atom_type=None):
        traj_data = []
        system_flatten = list(itertools.chain(*list(self.system.values())))
        masses_flatten = np.array([self.masses[atom] for atom in system_flatten])
        if atom_type != None:
            mask_atom = (np.array(system_flatten) == atom_type)
        if mol_type != None:
            from string import digits
            remove_digits = str.maketrans('', '', digits)
            mol_names = [s.translate(remove_digits) for s in list(self.system.keys())]
            mol_names = np.array(mol_names)
            n_atoms_per_mol = np.array(list(map(len, list(self.system.values()))))
            mol_names_allatom = np.repeat(mol_names, n_atoms_per_mol)
            mask_mol = np.array((mol_names_allatom == mol_type))
        with pytrr.GroTrrReader(traj_path) as trajectory:
            for step, frame in enumerate(trajectory):
                coordinates = []
                frame_data = trajectory.get_data()
                if step == 0:
                    box = np.array([frame_data['box'][i][i] for i in range(0, 3)])
                if mol_type != None:
                    traj_data.append(frame_data['x'] * np.tile(mask_mol, (3, 1)).swapaxes(1, 0))
                elif atom_type != None:
                    traj_data.append(frame_data['x'] * np.tile(mask_atom, (3, 1)).swapaxes(1, 0))
                else:
                    traj_data.append(frame_data['x'])
        traj_data = np.array(traj_data)
        bins = (np.array([6.273, 4.26]) // dx).astype(int)
        steps = traj_data.shape[0]
        traj_data = traj_data.reshape(-1, 3)
        hist_z, bins_z = np.histogram(traj_data[:, 2], bins=bins[0], range=[0.001, 6.272],
                                      weights=np.tile(masses_flatten, steps))
        volume_z = box[0] * box[1] * (bins_z[1] - bins_z[0])
        Da = 1.66053906660e-24  # Da in gramms
        nm_to_cm = 1e-7
        hist_z = hist_z * Da / volume_z / steps / nm_to_cm ** 3
        return hist_z, bins_z

    def calculate_orientational_order_1d(self, traj_path, dx, mol_type=None):
        pass