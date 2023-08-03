from rdkit import rdBase,Chem
import numpy as np
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from collections import Counter


sup = Chem.SDMolSupplier('tox21_10k_data_all.sdf')
train = [mol for mol in sup if mol is not None and 'NR-AR' in mol.GetPropsAsDict()]
true_list = [mol for mol in train if mol.GetPropsAsDict()['NR-AR']==1]
false_list = [mol for mol in train if mol.GetPropsAsDict()['NR-AR']==0]


max_display = 1000
img1 = Draw.MolsToGridImage(true_list[:max_display],molsPerRow=3, subImgSize=(200,200), legends=["true: " + str(x) for x in range(max_display)])

#ファイルに保存する
img1.save('true_grid.png')
max_display = 40
img2 = Draw.MolsToGridImage(false_list[:max_display],
                        molsPerRow=4, #一列に配置する分子の数
                        subImgSize=(400,400),
                        legends=["true: " + str(x) for x in range(max_display)]
)

#ファイルに保存する
img2.save('false_grid.png')

"""
bit_infos = []
for mol in true_list:
    bit_info = {}
    X = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, bitInfo=bit_info)
    bit_infos.append(bit_info)

print("bit_info_size: ", len(bit_info))
print("bit_infos_size: ", len(bit_infos))

#true_list : mol     bit_infos : bit_info
cnt = Counter()
for bit_info in bit_infos:
    for bit in list(bit_info.keys()):
        cnt[bit] += 1
#そのビットが立っている分子の数を数える
top10 = cnt.most_common(10)

print(top10)

selected_mols = []
selected_bit_info = []
n_most_common = 10
bits = [i for i,j in cnt.most_common(n_most_common)]
for (mol, bit_info) in zip(true_list, bit_infos):
    hit = 0
    for bit in bits:
        if bit in bit_info.keys():
            hit += 1
    #10条件を満たすと選ばれる
    if hit == len(bits):
        selected_mols.append(mol)
        selected_bit_info.append(bit_info)

print("共通ビット保有分子数：", len(selected_mols))

most_bit = cnt.most_common()[0][0]
print("most-bit:",most_bit)
target = selected_mols[0]
bit_info = selected_bit_info[0]
print("bits:", bits)
morgan_turples = ((target, bit, bit_info) for bit in bits)

img = Draw.DrawMorganBits(morgan_turples, molsPerRow=3, subImgSize=(800,800), legends=['bit: '+str(x) for x in range(n_most_common)])
print(type(img))
with open('sample.xml', mode='w') as f:
    f.write(img)


img2 = Draw.MolsToGridImage(selected_mols,
                        molsPerRow=4, #一列に配置する分子の数
                        subImgSize=(500,500),
                        legends=["selected: " + str(x) for x in range(len(selected_mols))]
)

#ファイルに保存する
img2.save('selected_grid.png')
"""