from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem, Draw, rdMHFPFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Sheridan
from rdkit.Avalon import pyAvalonTools
def inputData(fingerprint= "morgan"):
    sup = Chem.SDMolSupplier('tox21_10k_data_all.sdf')
    train = [mol for mol in sup if mol is not None and 'NR-AR' in mol.GetPropsAsDict()]
    if fingerprint == "morgan":
        X = [ AllChem.GetMorganFingerprintAsBitVect(m, 2) for m in train]
    elif fingerprint == "RDKit":
        X = [Chem.RDKFingerprint(m) for m in train]
    elif fingerprint == "MACCSKey":
        X = [AllChem.GetMACCSKeysFingerprint(m) for m in train]
    elif fingerprint == "rdMHFP":
        encoder=rdMHFPFingerprint.MHFPEncoder()
        X = [encoder.EncodeMol(m) for m in train]
    elif fingerprint == "Avalon":
        X = [pyAvalonTools.GetAvalonFP(m) for m in train]


    X = [ np.asarray(x) for x in X]
    #print(X)
    X = np.vstack(X)
    y = [m.GetIntProp('NR-AR') for m in train]
    y = np.array(y)

    # X:学習データ　　y:解答
    print('X:', X.shape, 'y:', y.shape)
    return X, y