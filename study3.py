from rdkit import rdBase,Chem
import numpy as np
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from collections import Counter


sup = Chem.SDMolSupplier('tox21_10k_data_all.sdf')
bitI_morgan = {}
train = [mol for mol in sup if mol is not None and 'NR-AR' in mol.GetPropsAsDict()]
true_list = [mol for mol in train if mol.GetPropsAsDict()['NR-AR']==1]
bit_infos = []
Xs = []
for mol in true_list:
    bit_info = {}
    X = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bit_info)
    bit_infos.append(bit_info)
    Xs.append(X)

cnt = Counter()
for bit_info in bit_infos:
    for bit in list(bit_info.keys()):
        cnt[bit] += 1

cnt.most_common(10)

print(cnt)

selected_mols = []
n_most_common = 10
bits = [i for i,j in cnt.most_common(n_most_common)]
for (mol, bit_info) in zip(true_list, bit_infos):
    hit = 0
    for bit in bits:
        if bit in bit_info.keys():
            hit += 1
    if hit == len(bits):
        selected_mols.append(mol)

print("共通ビット保有分子数：", len(selected_mols))
img = Draw.MolsToGridImage(selected_mols[:9], molsPerRow=3, legends=[x.GetProp('NR-AR') for x in selected_mols[:9]])
print(type(img))
img.save("NR-AR.png")

# 可視化　　https://future-chem.com/rdkit-fp-visualize/

