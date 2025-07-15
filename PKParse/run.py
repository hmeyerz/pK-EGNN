from utils import constants, load_train_dir
from pkparse import parser

import glob, os
from zipfile import ZipFile, ZIP_DEFLATED

import gzip


SRC_DIR = "../inputs"            # where *.bundle.npz already live
OUT_ZIP = "../pkegnn_dataset.zip"
MAP_DIR = "../res_maps"
FAIL_GZ = "../badparse.gz"



# --- train dict once on 200 random PDBs before the big loop ----------
#subset = random.sample(list(glob.glob("../pkegnn_inputs_test/*.npz")), k=200)
#train_zstd_dict(subset, DICT)

"""TODO: explain the maps 

array([[b'1A', b'1A'],
       [b'2A2', b'2A2'],
       [b'3A', b'3A'],
       [b'4A', b'4A'],
       [b'5A1', b'5A1'],
       [b'6A5', b'6A5'],
       
       How only titratable have a val after chain."""



import os, time
to=time.time()

#DATA_DIR = os.getcwd()
constants = constants() #obsolete neighbors
oldpdbs, newpdbs = load_train_dir()
#print(oldpdbs,newpdbs)
if not os.path.exists(SRC_DIR):
    os.mkdir(SRC_DIR)
with gzip.open(FAIL_GZ, "wb") as f:
    f.close()
if not os.path.exists(MAP_DIR):
    os.mkdir(MAP_DIR)


#if not os.path.exists(MAP_GZ):
#    os.mkdir(SRC_DIR)


print("working if files printed:",newpdbs[0],oldpdbs[0])
#print(newpdbs[0].split("gz"))
#print(newpdbs[0].split("gz")[0][-5:-1])
def run(num_pdbs="all"):
    """This takes the intersection of the fixed and original gzipped pdb files which are on disk and wraps them in a try-except clause."""
    t0 = time.time()
    if num_pdbs == "all":
        indices = len(oldpdbs)
    else:
        indices = num_pdbs

    for i in range(indices):
        try:
            parser(newpdbs[i], oldpdbs[i]).run()

        except Exception as e:
            pdb=newpdbs[i].split("gz")[-5:-1][0].encode()
            print(pdb)
            print("error in", pdb, ":", e)
            with gzip.open("../badparse.gz", "ab") as f:
                f.write(pdb)
                f.write(b"\n")
                continue

    print("zipping entire db",(time.time() - t0)/60, "mins")

    with ZipFile(OUT_ZIP, "w", compression=ZIP_DEFLATED) as zf:
        for path in glob.glob(os.path.join(SRC_DIR, "*.npz")):
            zf.write(path, arcname=os.path.basename(path))   # 1bjm.bundle.npz → ZIP
    print("Wrote", OUT_ZIP)
    print("data prep time:", (time.time() - t0)/60, "mins", indices, "pdbs")