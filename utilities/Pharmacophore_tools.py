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


def save_pharmacophore(pharmacophore: Pharm.BasicPharmacophore, path: str):
    # print("Saving Pharmacophore")
    writer = Pharm.FilePMLFeatureContainerWriter(path)
    writer.write(pharmacophore)
    writer.close()


def load_pml_pharmacophore(path):
    # print("Loading pharmacophore from %s" % path)
    ifs = Base.FileIOStream(path)
    r = Pharm.PMLPharmacophoreReader(ifs)
    pharm = Pharm.BasicPharmacophore()
    try:
        r.read(pharm)
        return pharm
    except:
        return False


def get_pharmacophore(mol: Chem.BasicMolecule, fuzzy=True) -> Pharm.BasicPharmacophore:
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
