#structure_files= glob.glob(path + "/structures/pdbs/*.gz") #+ glob.glob(path + "/structures/pdbs/*.pdb.gz")
import glob
import gzip
import os
from itertools import groupby
import numpy as np
import time

to=time.time()

elements = {
    b"D":1, b"H": 1, "LI":3, b"C": 6, b"N": 7, b"O": 8, 
    b"F":9,
    b"NA": 11, b"MG": 12,  b"P": 15, b"S": 16, b"CL": 17, b"K": 19,
    b"CA": 20, b"MN": 25, b"FE": 26, b"CO":27, b"NI":28,
    b"CU": 29, b"ZN": 30,
    b"SE":34,
    b"MO":42, b"SN":50, 
    b"I":53, b"CS":55,
    b"W":74, b"PT":78
    }

elements = {
    b"D":1, b"H": 1, "LI":3, b"C": 6, b"N": 7, b"O": 8, 
    b"F":9,
    b"NA": 11, b"MG": 12,  b"P": 15, b"S": 16, b"CL": 17, b"K": 19,
    b"CA": 20, b"MN": 25, b"FE": 26, b"CO":27, b"NI":28,
    b"CU": 29, b"ZN": 30,
    b"SE":34,
    b"MO":42, b"SN":50, 
    b"I":53, b"CS":55,
    b"W":74, b"PT":78
    }
code={b"HIS":b"0",
        b"ASP":b"1",
        b"LYS":b"2",
        b"TYR":b"3",
        b"GLU":b"4",
        b"CYS":b"5"}

target_files=glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/targets/*.npz")

def is_float(bs):
    if b'.' not in bs:
        return False
    else:
        return bs


def parsepdb(file):

    pdblines=[]

    with gzip.open(file, "r") as f:
        l=f.readlines()
        lines=np.char.array(f.readlines())
        f.close()

    lines=[line for line in l if np.char.startswith(line,b"ATOM")]
    

    for i,l in enumerate(lines):
        l=l.split()

        #resname 
        info=l[3] 
        if info in code.keys(): #titratable
            aline=[code[info]]
        else:
            continue
        
        #atomnumber
        try:
            info=elements[l[-1].strip(b"+-0123456789")]
        except:
            try:
                info=elements[l[2]]
            except:
                continue
        aline.append(info)

        #coors
        xyz=[]
        info=(l[6],l[7],l[8])
        for c in info:
            if is_float(c):
                xyz.append(c)
        
        if len(xyz) < 3:
            continue
        else:
            aline.append(xyz[0])
            aline.append(xyz[1])
            aline.append(xyz[2])

        #chain   
        chain=l[4]
        if is_float(chain):
            continue
        else:
            aline.append(chain) 

        #resnum
        resnum=l[5]
        if is_float(resnum):
            continue
        else:
            aline.append(resnum.strip(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")) 


        pdblines.append(aline)
        
    #whole pdb
    pdblines = [list(group) for key, group in groupby(pdblines, key=lambda row: row[-1])]
    
    #myinfo
    zs=np.array([[np.array([np.uint8(l[1]) for l in resi])] for resi in pdblines],dtype=object)
    coors=np.array([np.array([[np.float32(l[2]),np.float32(l[3]),np.float32(l[4])] for l in resi]) for resi in pdblines],dtype=object) #TODO it saved as normal float
    pdblines=[r[0] for r in pdblines]
    ids=np.char.array([np.char.add(np.char.add(l[-1],l[-2]),l[0]) for l in pdblines])

    #return
    if pdblines:
        return ids, zs, coors
    else:
        return None, None, None

    

    #else:
        #continue

failparse=[]
for file in target_files:
    try:
        pdb=file[-8:-4]
        array=np.load(file)
        pks,pids = array["pks"], np.char.array(array["ids"])
        
        path=f"/home/jrhoernschemeyer/Desktop/data_prep/structures/pdbs/{pdb}.pdb.gz"
        if os.path.exists(path):
            mids, zs, coors =parsepdb(path)
            if type(mids) != None and type(pids) != None:
                common_vals, pidx, midx = np.intersect1d(pids, mids,return_indices=True)
                ids=pids[pidx]
                pks=pks[pidx]
                pos=coors[midx]
                species=zs[midx]
                np.savez_compressed(f"/home/jrhoernschemeyer/Desktop/data_prep/inputs/{pdb}.npz",z=species, pos=pos, pks=pks,ids=ids)
                
                
            else:
                failparse.append(pdb)
    except:
        failparse.append(pdb)
       
with gzip.open("/home/jrhoernschemeyer/Desktop/data_prep/failparse_final.gz","wb") as f:
    for pdb in failparse:
        f.write(pdb.encode())
        f.write(b" ")


print("done 20k",(time.time() - to)/60,"mins")


to=time.time()

paths=glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/nometals/inputs/*")
paths=np.array(paths)
val=paths[np.random.random_integers(low=0,high=len(paths)-2,size=np.int(0.4*len(paths)))]

train=np.char.array(list(set(paths).difference(set(val))))
np.savez_compressed("/home/jrhoernschemeyer/Desktop/data_prep/split.npz",val=val, train=train)