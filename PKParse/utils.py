from sklearn.neighbors import NearestNeighbors
import numpy as np
import glob
import os

#data directories
rcsb_paths     =np.char.array(glob.glob("../../data/pdbs/rcsb/*.gz"))
modeled_paths   =np.char.array(glob.glob("../../data/pdbs/fixed/*.gz"))

data_dir = os.getcwd()

fullcode = {
    b"HIS":b"0", b"ASP":b"1", b"LYS":b"2",
    b"TYR":b"3", b"GLU":b"4", b"CYS":b"5"
}
code = {
    b"H":b"0", b"A":b"1", b"L":b"2",
    b"T":b"3", b"G":b"4", b"C":b"5"
}
amber = {
    0:b"OE2", 1:b"OD2", 2:b"SG",
    3:b"OH",  4:b"NZ",  5:b"ND1"
}

three2one = {
    b"ALA":"A", b"ARG":"R", b"ASN":"N", b"ASP":"D", b"CYS":"C",
    b"GLU":"E", b"GLN":"Q", b"GLY":"G", b"HIS":"H", b"ILE":"I",
    b"LEU":"L", b"LYS":"K", b"MET":"M", b"PHE":"F", b"PRO":"P",
    b"SER":"S", b"THR":"T", b"TRP":"W", b"TYR":"Y", b"VAL":"V"
}
# Complete periodic table (IUPAC symbols) + deuterium, as bytes → atomic number #debug
cofactors = {
    b"D":   1,   b"H":   1,
    b"He":  2,   b"Li":  3,   b"Be":  4,   b"B":   5,   b"C":   6,   b"N":   7,
    b"O":   8,   b"F":   9,   b"Ne": 10,   b"Na": 11,   b"Mg": 12,   b"Al": 13,
    b"Si": 14,   b"P":  15,   b"S":  16,   b"Cl": 17,   b"Ar": 18,   b"K":  19,
    b"Ca": 20,   b"Sc": 21,   b"Ti": 22,   b"V":  23,   b"Cr": 24,   b"Mn": 25,
    b"Fe": 26,   b"Co": 27,   b"Ni": 28,   b"Cu": 29,   b"Zn": 30,   b"Ga": 31,
    b"Ge": 32,   b"As": 33,   b"Se": 34,   b"Br": 35,   b"Kr": 36,   b"Rb": 37,
    b"Sr": 38,   b"Y":  39,   b"Zr": 40,   b"Nb": 41,   b"Mo": 42,   b"Tc": 43,
    b"Ru": 44,   b"Rh": 45,   b"Pd": 46,   b"Ag": 47,   b"Cd": 48,   b"In": 49,
    b"Sn": 50,   b"Sb": 51,   b"Te": 52,   b"I":  53,   b"Xe": 54,   b"Cs": 55,
    b"Ba": 56,   b"La": 57,   b"Ce": 58,   b"Pr": 59,   b"Nd": 60,   b"Pm": 61,
    b"Sm": 62,   b"Eu": 63,   b"Gd": 64,   b"Tb": 65,   b"Dy": 66,   b"Ho": 67,
    b"Er": 68,   b"Tm": 69,   b"Yb": 70,   b"Lu": 71,   b"Hf": 72,   b"Ta": 73,
    b"W":  74,   b"Re": 75,   b"Os": 76,   b"Ir": 77,   b"Pt": 78,   b"Au": 79,
    b"Hg": 80,   b"Tl": 81,   b"Pb": 82,   b"Bi": 83,   b"Po": 84,   b"At": 85,
    b"Rn": 86,   b"Fr": 87,   b"Ra": 88,   b"Ac": 89,   b"Th": 90,   b"Pa": 91,
    b"U":  92,   b"Np": 93,   b"Pu": 94,   b"Am": 95,   b"Cm": 96,   b"Bk": 97,
    b"Cf": 98,   b"Es": 99,   b"Fm":100,   b"Md":101,   b"No":102,   b"Lr":103,
    b"Rf":104,   b"Db":105,   b"Sg":106,   b"Bh":107,   b"Hs":108,   b"Mt":109,
    b"Ds":110,   b"Rg":111,   b"Cn":112,   b"Nh":113,   b"Fl":114,   b"Mc":115,
    b"Lv":116,   b"Ts":117,   b"Og":118,
    #get the ones with symbols
    b"Zn2+": 30,
    b"Na1+": 11,
    b"Be2+": 4,
    b"Ca2+":20
}

elements = {b"C":   6,   b"N":   7, b"F":9,
    b"O":   8, b"P":  15, b"S":  16} #debug2: add F and P since it is in one pdb. #TODO: which pdb!?



def constants():
    """use because pkparser is per pdb."""
    ligands = np.load(data_dir + "/data/ligands.npz")["data"]
    #nbrs = NearestNeighbors(n_neighbors=n_neighbors, algorithm="brute")
    cnbrs = NearestNeighbors(algorithm="brute") #TODO
    byte_constants = {"titratables_full": fullcode,
                 "titratables_short": code,
                 "aa_code": three2one,
                 "elements": elements,
                 "cofactors": cofactors,
                 "ligands": ligands,
                 "disulfide_nn": cnbrs,
                 "amber_sites": amber
    }
    return byte_constants

def load_train_dir():
    """returns the strings for the intersection of the old pdbs (rcsb) and the new pdbs (modeled)"""
    pdbs=rcsb_paths
    fixed_pdbs=modeled_paths
    _, og, mdled = np.intersect1d(np.char.array([f[-11:-7] for f in pdbs]),np.char.array([f[-7:-3] for f in fixed_pdbs]), return_indices=True)
    return pdbs[og], fixed_pdbs[mdled]



