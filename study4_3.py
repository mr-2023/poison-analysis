from rdkit import rdBase,Chem
import numpy as np
from rdkit.Chem import AllChem, Draw
from rdkit import DataStructs
from collections import Counter


sup = Chem.SDMolSupplier('tox21_10k_data_all.sdf')
train = [mol for mol in sup if mol is not None and 'NR-AR' in mol.GetPropsAsDict()]
true_list = [mol for mol in train if mol.GetPropsAsDict()['NR-AR']==1]
false_list = [mol for mol in train if mol.GetPropsAsDict()['NR-AR']==0]


max_display = 60
img1 = Draw.MolsToGridImage(true_list[:max_display],molsPerRow=3, subImgSize=(500,500), legends=["true: " + str(x) for x in range(max_display)])

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

target = true_list[0]
bit_info={}
X=Chem.RDKFingerprint(target, bitInfo=bit_info)

print("bit_info_size: ", len(bit_info))
i=0
while not i in bit_info.keys():
    i+=1
print("bit_info[0]: ", bit_info[i])


morgan_turples = ((target, bit, bit_info) for bit in list(bit_info.keys())[:12])
img = Draw.DrawRDKitBits(morgan_turples, molsPerRow=5, legends=['bit: '+str(x) for x in list(bit_info.keys())[:12]])

with open('sample_rdkit_test.xml', mode='w') as f:
    f.write(img)