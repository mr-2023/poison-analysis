from rdkit import Chem
from rdkit.Chem import Draw
sup = Chem.SDMolSupplier('tox21_10k_data_all.sdf')
train = [mol for mol in sup
         if mol is not None and 'NR-AR' in mol.GetPropsAsDict()]
print(train[0].GetPropsAsDict())
output = False
count=0

for num in range(len(train)):
    if output:
        print("----------"+str(num)+"----------------")
    target = train[num]
    if output:
        print(target.GetPropsAsDict())
        print("Formula:" + target.GetPropsAsDict()["Formula"])
        print("FW:"+ str(target.GetPropsAsDict()["FW"]))
        print("DSSTox_CID:" + str(target.GetPropsAsDict()["DSSTox_CID"]))
    frag = 0
    if target.GetPropsAsDict()["NR-AR"]==0:
        exist_NR_AR = "なし"
    else:
        exist_NR_AR = "あり"
        frag = 1
    if output:
        print("NR-AR:" + exist_NR_AR)
    if frag == 1:
        Draw.MolToFile(target, './truePage/test-{}.png'.format(count), size=(300,300))
        count+=1
