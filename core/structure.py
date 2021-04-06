import numpy as np

class System:
    """
    general class for system data structure
    """
    def __init__(self, atomlist, mollist=None, indexes=None):
        self.atomlist = atomlist
        self.mollist = mollist
        self.indexes = indexes

class Trajectory:
    """
    general class for trajectory data structure
    """

    def __init__(self, box, coordinates):
        self.box = box
        self.coordinates = coordinates

    def recalculate_to_com(self, molecule_list):
        pass