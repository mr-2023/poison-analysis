from rdkit import rdBase,Chem
import numpy as np
from rdkit.Chem import AllChem, Draw
import requests
from IPython.display import SVG

sup = Chem.SDMolSupplier('tox21_10k_data_all.sdf')
bitI_morgan = {}
train = [mol for mol in sup if mol is not None and 'NR-AR' in mol.GetPropsAsDict()]
true_list = [mol for mol in train if mol.GetPropsAsDict()['NR-AR']==1]
target = train[0]

fp_morgan = AllChem.GetMorganFingerprintAsBitVect(target, 2, bitInfo=bitI_morgan)

print('Morgan Fingerprintのビット数:', fp_morgan.GetNumBits())

for key in list(bitI_morgan.keys())[:5]:
    print('ビット:{};, 関連する部分構造:{}'.format(key, bitI_morgan[key]))

morgan_turples = ((target, bit, bitI_morgan) for bit in list(bitI_morgan.keys())[:12])
img = Draw.DrawMorganBits(morgan_turples, molsPerRow=3,legends=['bit: '+str(x) for x in list(bitI_morgan.keys())[:12]],subImgSize=(800,800))
print(type(img))
with open('sample.xml', mode='w') as f:
    f.write(img)