import CDPL.Chem as Chem
import CDPL.Base as Base
import CDPL.Biomol as Biomol
import CDPL.Pharm as Pharm
from argparse import ArgumentParser
from typing import Dict, List, Set, Tuple
import os

from src.pharmacophore_tools import getPharmacophore, savePharmacophore, getInteractionPharmacophore
from src.molecule_tools import sanitize_mol


COFACTOR_LIGANDS = ['HEM', 'NAG', 'FUC', 'ATP', 'ADP', 'AMP', 'PEG', 'PGF', 'MAL', 'PTR', 'FUL', 'OCS', 'PHE',
                    'NDP', 'MES', 'ANP', 'JZR']

THREE_LETTER_AMINO_ACID_CODES = {
    'ALA',
    'ARG',
    'ASN',
    'ASP',
    'ASX',  # asparagine or aspartic acid
    'CYS',
    'GLU',
    'GLN',
    'GLX',  # glutamine or glutamic acid
    'GLY',
    'HIS',
    'ILE',
    'LEU',
    'LYS',
    'MET',
    'PHE',
    'PRO',
    'SER',
    'THR',
    'TRP',
    'TYR',
    'VAL'
}


def parseArgs():
    parser = ArgumentParser()
    parser.add_argument('-i', required=True, type=str, help='Input pdb file')
    parser.add_argument('-o', required=True, type=str, help='output folder')
    parser.add_argument('-ligandCodes', required=False, default=None, type=str, help='comma separated list of ligand codes')
    parser.add_argument('-notFuzzy', required=False, default=False, action='store_true', help='generate non-fuzzy pharmacophores')
    parser.add_argument('-notXvols', required=False, default=False, action='store_true', help='omit exclusion volumes')
    args = parser.parse_args()
    return args


def readPDBFromStream(stream: Base.IOStream):
    r = Biomol.PDBMoleculeReader(stream)
    p = Chem.BasicMolecule()
    r.read(p)
    return p


def readPDBFromFile(path: str) -> Chem.BasicMolecule:
    s = Base.FileIOStream(path)
    protein = readPDBFromStream(s)
    sanitize_mol(protein)
    return protein


def extractLigands(protein: Chem.BasicMolecule, ligandCodes: Set = None) -> Dict[str, Chem.BasicMolecule]:
    extractedLigands = {}

    atomsToRemove = set()
    # bondsToRemove = set()
    for i in range(protein.numAtoms):
        if i in atomsToRemove:
            continue

        a = protein.getAtom(i)
        if isProteinResidue(a):
            continue

        elif isWater(a):
            atomsToRemove.add(i)
            continue

        elif Chem.isMetal(a):
            atomsToRemove.add(i)
            # for b in a.bonds:
            #     bIndex = protein.getBondIndex(b)
            #     if bIndex not in bondsToRemove:
            #         bondsToRemove.add(bIndex)
            # continue

        else:  # atom must be some form of ligand, since it is none of the above -> careful with peptide ligands!!!
            ligandCode = str(Biomol.getResidueCode(a))
            ligandAtomIndices = findLigand(protein, ligandCode)
            if isLigand(a, ligandCodes):
                extractedLigands[ligandCode] = createLigandFromAtoms(protein, [protein.getAtom(aIndex) for aIndex in ligandAtomIndices])

            # atom is some form of ligand but not provided in list --> remove
            for aIndex in ligandAtomIndices:
                if aIndex not in atomsToRemove:
                    atomsToRemove.add(aIndex)
            continue

    atomsToRemove = list(atomsToRemove)
    # bondsToRemove = list(bondsToRemove)
    atomsToRemove.sort(reverse=True)
    # bondsToRemove.sort(reverse=True)

    # remove atoms / bonds inplace
    for aIndex in atomsToRemove:
        protein.removeAtom(aIndex)
    # for bIndex in bondsToRemove:
    #     protein.removeBond(bIndex)

    return extractedLigands


def isWater(atom: Chem.BasicAtom) -> bool:
    return Biomol.getResidueCode(atom) == 'HOH'


def isProteinResidue(atom: Chem.BasicAtom) -> bool:
    return Biomol.getResidueCode(atom) in THREE_LETTER_AMINO_ACID_CODES


def isLigand(atom: Chem.BasicAtom, ligandCodes: Set = None) -> bool:
    if ligandCodes is not None:
        return Biomol.getResidueCode(atom) in ligandCodes


def findLigand(protein: Chem.BasicMolecule, ligandCode: str) -> List[int]:
    ligandIndices = [i for i in range(protein.numAtoms) if Biomol.getResidueCode(protein.getAtom(i)) == ligandCode]
    return ligandIndices


def createLigandFromAtoms(protein: Chem.BasicMolecule, atoms: List[Chem.BasicAtom]) -> Chem.BasicMolecule:
    frag = Chem.Fragment()
    addedBonds = set()
    for a in atoms:
        for b in a.bonds:
            if protein.getBondIndex(b) in addedBonds:
                continue

            if isProteinResidue(b.getBegin()) or isProteinResidue(b.getEnd()):
                continue  # covalent bond

            frag.addBond(b)
    frag = Chem.perceiveComponents(frag)[0]
    Chem.perceiveSSSR(frag, False)

    ligand = Chem.BasicMolecule()
    ligand.assign(frag)
    sanitize_mol(ligand)
    return ligand


def processPDBStructure(protein: Chem.BasicMolecule,
                        outputFolder: str,
                        ligandCodes: Set = None,
                        fuzzy: bool = True,
                        exclusionVolumes: bool = True
                        ) -> Tuple[Chem.BasicMolecule,
                                   Dict[str, Chem.BasicMolecule],
                                   Dict[str, Pharm.BasicPharmacophore],
                                   Dict[str, Pharm.BasicPharmacophore]]:

    # clean protein and extract ligands
    extractedLigands = extractLigands(protein, ligandCodes=ligandCodes)
    if len(extractedLigands) == 0:
        print('No ligand found with given ligand-codes: ', ligandCodes)
        return protein, extractedLigands, {}, {}

    # check whether we still have a sufficiently large protein structure
    if protein.numAtoms < 150:
        print('Protein has less than 150 Atoms after cleaning -> skipping')
        return protein, extractedLigands, {}, {}

    interactionPharmacophores = {}
    ligandPharmacophores = {}
    for ligandCode, ligand in extractedLigands.items():
        interactionPharmacophore = getInteractionPharmacophore(protein, ligand, fuzzy=fuzzy, exclusionVolumes=exclusionVolumes)
        interactionPharmacophores[ligandCode] = interactionPharmacophore
        ligandPharmacophore = getPharmacophore(ligand, fuzzy=fuzzy)
        ligandPharmacophores[ligandCode] = ligandPharmacophore

        if not os.path.isdir(outputFolder):
            os.mkdir(outputFolder)
        savePharmacophore(interactionPharmacophore, '{}/interaction-pharmacophore.pml'.format(outputFolder))
        savePharmacophore(ligandPharmacophore, '{}/ligand-pharmacophore.pml'.format(outputFolder))

    return protein, extractedLigands, ligandPharmacophores, interactionPharmacophores


def main():
    args = parseArgs()
    protein = readPDBFromFile(args.i)
    ligandCodes = args.ligandCodes.split(',') if args.ligandCodes is not None else set()
    processPDBStructure(protein,
                        args.o,
                        ligandCodes=ligandCodes,
                        fuzzy=(not args.notFuzzy),
                        exclusionVolumes=(not args.notXvols)
                        )


if __name__ == '__main__':
    main()
