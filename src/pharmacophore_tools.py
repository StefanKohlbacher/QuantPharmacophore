from typing import List, Union, Dict, Tuple

import CDPL.Pharm as Pharm
import CDPL.Base as Base
import CDPL.Chem as Chem


FEATURE_TYPES = {
    1: Pharm.FeatureType.HYDROPHOBIC,
    2: Pharm.FeatureType.AROMATIC,
    3: Pharm.FeatureType.NEG_IONIZABLE,
    4: Pharm.FeatureType.POS_IONIZABLE,
    5: Pharm.FeatureType.H_BOND_DONOR,
    6: Pharm.FeatureType.H_BOND_ACCEPTOR,
    7: Pharm.FeatureType.X_VOLUME,
}
FEATURE_TYPES_INVERSE = {value: key for key, value in FEATURE_TYPES.items()}


def savePharmacophore(pharmacophore: Pharm.BasicPharmacophore, path: str) -> None:
    # print("Saving Pharmacophore")
    writer = Pharm.FilePMLFeatureContainerWriter(path)
    writer.write(pharmacophore)
    writer.close()


def loadPharmacophore(path: str) -> Union[Pharm.BasicPharmacophore, None]:
    # print("Loading pharmacophore from %s" % path)
    ifs = Base.FileIOStream(path)
    r = Pharm.PMLPharmacophoreReader(ifs)
    pharm = Pharm.BasicPharmacophore()
    try:
        r.read(pharm)
        return pharm
    except:
        return None


def getPharmacophore(mol: Chem.BasicMolecule, fuzzy=True) -> Pharm.BasicPharmacophore:
    """

    :param mol: Molecule to generate pharmacophore from.
    :param fuzzy: Indicates whether to generate a vector for HBD and HBA or just a sphere. Fuzzy=sphere.
    :return:
    """
    if isinstance(mol, Pharm.BasicPharmacophore):
        return mol
    assert isinstance(mol, Chem.BasicMolecule), "Given object should be of type Chem.BasicMolecule, %s was given" % type(mol)
    Pharm.prepareForPharmacophoreGeneration(mol)  # Fails silently if molecule has coordinates == 0 !!!
    # Chem.makeHydrogenComplete(mol)
    Chem.generateHydrogen3DCoordinates(mol, False)
    pharm = Pharm.BasicPharmacophore()
    pharm_generator = Pharm.DefaultPharmacophoreGenerator(fuzzy)
    pharm_generator.generate(mol, pharm)
    if fuzzy:
        for f in pharm:
            if Pharm.getType(f) == 5 or Pharm.getType(f) == 6:
                Pharm.clearOrientation(f)
                Pharm.setGeometry(f, Pharm.FeatureGeometry.SPHERE)
    return pharm


def getInteractionPharmacophore(protein: Chem.BasicMolecule,
                                ligand: Chem.BasicMolecule,
                                exclusionVolumes=True,
                                fuzzy=False,
                                ) -> Pharm.BasicPharmacophore:
    assert isinstance(protein, Chem.BasicMolecule) and isinstance(ligand, Chem.BasicMolecule)
    Chem.clearSSSR(protein)
    Pharm.prepareForPharmacophoreGeneration(protein)
    Pharm.prepareForPharmacophoreGeneration(ligand)

    int_pharm = Pharm.BasicPharmacophore()
    pharm_gen = Pharm.InteractionPharmacophoreGenerator(fuzzy, True)  # fuzzy core ph4, fuzzy env. ph4
    pharm_gen.addExclusionVolumes(exclusionVolumes)
    pharm_gen.generate(ligand, protein, int_pharm, True)  # True means ligand environment shall be extracted first

    if fuzzy:
        for f in int_pharm:
            if Pharm.getType(f) == 5 or Pharm.getType(f) == 6:
                Pharm.clearOrientation(f)
                Pharm.setGeometry(f, Pharm.FeatureGeometry.SPHERE)
    return int_pharm
