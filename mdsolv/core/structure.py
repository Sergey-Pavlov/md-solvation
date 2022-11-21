import numpy as np

class System:
    """
    general class for system data structure
    """
    def __init__(self, atomlist, mollist=None, indexes=None, box=None, coordinates=None):
        self.atomlist = atomlist
        self.mollist = mollist
        self.indexes = indexes
        self.coordinates = coordinates
        self.box = box

    def __add__(self, other):
        atomlist = self.atomlist + other.atomlist
        mollist = self.mollist + other.mollist
        last_index = self.indexes[-1][-1]
        indexes = self.indexes.copy()
        for ind in other.indexes:
            indexes.append(np.array(ind) + last_index)
        box = self.box
        if self.coordinates is None or other.coordinates is None:
            return System(atomlist, mollist, indexes)
        else:
            coordinates = np.concatenate((self.coordinates, other.coordinates), axis=0)
            return System(atomlist, mollist, indexes, box, coordinates)

class Trajectory:
    """
    general class for trajectory data structure
    """

    def __init__(self, box, coordinates):
        self.box = box
        self.coordinates = coordinates

    def recalculate_to_com(self, molecule_list):
        pass