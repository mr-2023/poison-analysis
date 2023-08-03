from rdkit import Chem
import numpy as np
from rdkit.Chem import AllChem, Draw, rdMHFPFingerprint
from rdkit.Chem.AtomPairs import Pairs
from rdkit.Chem.AtomPairs import Sheridan
from rdkit.Avalon import pyAvalonTools
def failPicture(y_test,y_test_pred):
    sup = Chem.SDMolSupplier('tox21_10k_data_all.sdf')
    train = [mol for mol in sup if mol is not None and 'NR-AR' in mol.GetPropsAsDict()]
    count=0
    for i in range(len(y_test)):
        if y_test[i] != y_test_pred[i]:
            Draw.MolToFile(train[i], './failPage/test-{}.png'.format(count), size=(300,300))
            count+=1