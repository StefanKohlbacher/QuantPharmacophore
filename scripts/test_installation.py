# make some import so we can verify that we installed the packages correctly

try:
    import CDPL.Chem as Chem
    import CDPL.Base as Base
    import CDPL.Pharm as Pharm
    import CDPL.Math as Math

    print('Successfully loaded CDPL packages')
except ImportError:
    print('Failed to import CDPL packages')

try:
    import numpy as np
    import pandas as pd
    import sklearn as sk
    import scipy as sp
    import matplotlib.pyplot as plt

    print('Successfully loaded numpy, pandas, scikit-learn, scipy, matplotlib')
except ImportError:
    print('Failed to import standard python libraries')

import os
print('Current working directory', os.getcwd())

try:
    from src.hyperpharmacophore import DistanceHyperpharmacophore
    from src.molecule_tools import SDFReader
    from src.modules import loadMolecules, splitSamplesActivities, DEFAULT_TRAINING_PARAMETERS
    from src.utils import selectMostRigidMolecule

    print('Successfully import custom packages')

    print('Testing simple model initialization')

    if os.path.isfile('./_test/acetyl_conformations.sdf'):
        molecules = loadMolecules('./_test/acetyl_conformations.sdf')
        molecules, activities = splitSamplesActivities(molecules, 'pchembl_value')

        template, remainingMolecules = selectMostRigidMolecule(molecules)
        model = DistanceHyperpharmacophore([template, remainingMolecules[0]], **DEFAULT_TRAINING_PARAMETERS)

    else:
        model = DistanceHyperpharmacophore()

    print('Successfully initialized model')

except ImportError:
    print('Failed to import custom modules')

print('Finished import _test')
