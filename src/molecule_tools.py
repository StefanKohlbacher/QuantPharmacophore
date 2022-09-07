import CDPL.Base as Base
import CDPL.Chem as Chem
import CDPL.Math as Math
import CDPL.Pharm as Pharm
from collections import Iterable
import os
from src.pharmacophore_tools import getPharmacophore


ALLOWED_ATOMS = [1, 6, 7, 8, 9, 15, 16, 17, 35, 53]
SALT_METALS = [3, 11, 12, 19, 20]


class SDFReader:

    """
    Handles reading molecules from SDF files and takes care of necessary preparation steps.
    """

    def __init__(self, path: str, multiconf: bool=True, nr_mols: int=-1, properties: list=None):
        """

        :param path:
        :param multiconf:
        :param nr_mols:
        :param properties: List of properties to be read from SDF file. Returns None if not found. Internally adds ' <'
        and '>' before and after the property names in order to comply with the toolkit.
        """
        if not os.path.isfile(path):
            raise IOError('{} not found.'.format(path))
        if not path.endswith('.sdf'):
            raise IOError('Invalid file. Please provide an .sdf file. {} was given'.format(path))

        self.r = Chem.FileSDFMoleculeReader(path)
        self.multiconf = multiconf
        self.nr_mols = nr_mols
        self.nr_samples = None
        self.properties = set([" <"+p+">" for p in properties]) if properties is not None else None

        Chem.setMultiConfImportParameter(self.r, multiconf)
        if multiconf:
            Chem.setMultiConfInputProcessorParameter(self.r, Chem.DefaultMultiConfMoleculeInputProcessor(False,
                                                                                                        Chem.AtomPropertyFlag.TYPE | Chem.AtomPropertyFlag.ISOTOPE | Chem.AtomPropertyFlag.FORMAL_CHARGE,
                                                                                                        Chem.BondPropertyFlag.ORDER))
        self._gen = iter(self)

    def __len__(self):
        if self.nr_samples is None:
            self.nr_samples = self.r.getNumRecords()
            return self.nr_samples
        return self.nr_samples

    def __iter__(self):
        if self.properties is None:
            i = 0
            while True:
                mol = Chem.BasicMolecule()
                try:
                    if self.r.read(mol):
                        yield sanitize_mol(mol)
                    else:
                        break
                except IOError:
                    yield None

                i += 1
                if i == self.nr_mols:
                    break
        else:
            i = 0
            while True:
                mol = Chem.BasicMolecule()
                try:
                    if self.r.read(mol):
                        read_properties = self._extract_properties_from_mol(mol)
                        yield sanitize_mol(mol), read_properties
                    else:
                        break
                except IOError:
                    yield None

                i += 1
                if i == self.nr_mols:
                    break

    def __call__(self):
        return iter(self)

    def _extract_properties_from_mol(self, mol):
        read_properties = {}
        data = Chem.getStructureData(mol)
        for element in data:
            if element.header in self.properties:
                read_properties[element.header[2:-1]] = element.data
        return read_properties

    def read_all(self):
        """
        Reads all the molecules from the SDF file with set properties
        :return:
        """
        mols = {}
        for i, mol in enumerate(self):
            name = Chem.getName(mol)
            if len(name) == 0:  # no name set
                name = str(i)
            mols[name] = mol
        return mols

    def read(self):
        try:
            return next(self._gen)
        except IOError:
            return None


def remove_metal_salts(mol: Chem.BasicMolecule) -> Chem.BasicMolecule:
    to_remove = []
    for atom in mol.atoms:
        if Chem.getType(atom) not in SALT_METALS:
            continue
        else:
            to_remove.append(mol.getAtomIndex(atom))
    to_remove.sort()
    to_remove.reverse()
    for index in to_remove:
        mol.removeAtom(index)
    return mol


def is_metal(mol: Chem.BasicMolecule) -> bool:
    """
    Indicate if the compound contains a metal
    """
    for atom in mol.atoms:
        if Chem.getType(atom) in ALLOWED_ATOMS:
            continue
        else:
            return True
    return False


def remove_components(mol: Chem.BasicMolecule) -> Chem.BasicMolecule:
    components = Chem.getComponents(mol)
    largest_component = None  # set default number of atoms and index
    for comp in components:
        if largest_component is None:
            largest_component = comp
        elif comp.numAtoms > largest_component.numAtoms:
            largest_component = comp
    new_mol = Chem.BasicMolecule()
    new_mol.assign(largest_component)
    if Chem.hasStructureData(mol):
        Chem.setStructureData(new_mol, Chem.getStructureData(mol))
    return new_mol


def is_inorganic(mol: Chem.BasicMolecule) -> bool:
    for atom in mol.atoms:
        if Chem.getType(atom) != 6:
            continue
        else:
            return False
    return True


def neutralise(mol: Chem.BasicMolecule) -> Chem.BasicMolecule:
    to_remove = []
    for atom in mol.atoms:
        if Chem.getFormalCharge(atom) != 0:
            form_charge = Chem.getFormalCharge(atom)

            if form_charge != 0:
                for nbr_atom in atom.atoms:
                    if Chem.getFormalCharge(nbr_atom) != 0:
                        form_charge = 0
                        break  # it's fine if neighbor is charged too -> we assume it's the opposite charge

            if form_charge != 0:
                if form_charge > 0:
                    form_charge -= Chem.getImplicitHydrogenCount(atom)

                    if form_charge < 0:  # if charge is negative we set to zero and calculate the number of hydrogens later on
                        form_charge = 0

                    for nbr_atom in atom.atoms:
                        if form_charge == 0:
                            break

                        if Chem.getType(nbr_atom) == Chem.AtomType.H:
                            to_remove.append(mol.getAtomIndex(nbr_atom))
                            form_charge -= 1

                    Chem.setFormalCharge(atom, form_charge)

                else:
                    Chem.setFormalCharge(atom, 0)

    if len(to_remove) > 0:
        to_remove.sort()
        to_remove.reverse()
        for index in to_remove:
            mol.removeAtom(index)

        for atom in mol.atoms:
            Chem.setImplicitHydrogenCount(atom, Chem.calcImplicitHydrogenCount(atom, mol))
            Chem.setHybridizationState(atom, Chem.perceiveHybridizationState(atom, mol))
    return mol


def sanitize_mol(mol: Chem.BasicMolecule) -> Chem.BasicMolecule:
    Chem.calcImplicitHydrogenCounts(mol, True)
    Chem.perceiveHybridizationStates(mol, True)
    Chem.perceiveComponents(mol, True)
    Chem.perceiveSSSR(mol, True)
    Chem.setRingFlags(mol, True)
    Chem.setAromaticityFlags(mol, True)
    # Chem.makeHydrogenComplete(mol)
    # Chem.calcImplicitHydrogenCounts(mol, True)
    return mol


def clean(mol: Chem.BasicMolecule) -> "bool or Chem.BasicMolecule":
    """
    Checks if the molecule is a metal or inorganic -> return False
    Removes salts [Na+, Mg2+, Ca2+, K+, Li+] and multiple components -> keep the largest component.
    Neutralise molecule by adding or removing protons as well as possible.
    :param mol:
    :return:
    """
    mol = sanitize_mol(mol)

    # clean molecule
    mol = remove_metal_salts(mol)
    if is_metal(mol):
        return False
    if Chem.getComponentCount(mol) > 1:
        mol = remove_components(mol)
    if is_inorganic(mol):
        return False
    mol = neutralise(mol)
    return sanitize_mol(mol)


def mol_to_sdf(molecules, path, multiconf=True):
    if not isinstance(molecules, Iterable):
        molecules = [molecules]
    w = Chem.FileSDFMolecularGraphWriter(path)
    Chem.setMultiConfExportParameter(w, multiconf)
    for mol in molecules:
        Chem.calcImplicitHydrogenCounts(mol, False)
        w.write(mol)
    w.close()


def calculateStandardProperties(mol):
    standardProperties = {
        'nrAcceptors': [],
        'nrDonors': [],
        # 'nrRings': [],
        'nrRotBonds': [],
        'molWeight': [],
        'nrHeavyAtoms': [],
        'cLogP': [],
        'TPSA': [],
    }

    try:
        iter(mol)
    except:
        mol = [mol]

    for m in mol:
        Chem.calcTopologicalDistanceMatrix(m, True)

        p = getPharmacophore(m)
        hba, hbd = 0, 0
        for f in p:
            if Pharm.getType(f) == Pharm.FeatureType.H_BOND_ACCEPTOR:
                hba += 1
            elif Pharm.getType(f) == Pharm.FeatureType.H_BOND_DONOR:
                hbd += 1

        standardProperties['nrAcceptors'].append(hba)
        standardProperties['nrDonors'].append(hbd)
        standardProperties['molWeight'].append(Chem.calcExplicitMass(m))
        standardProperties['nrHeavyAtoms'].append(Chem.getHeavyAtomCount(m))
        standardProperties['cLogP'].append(Chem.calcXLogP(m))
        standardProperties['TPSA'].append(Chem.calcTPSA(m))
        standardProperties['nrRotBonds'].append(Chem.getRotatableBondCount(m, False, False))

    return standardProperties
