import numpy as np
import gzip
import time
import glob
#from sklearn.neighbors import NearestNeighbors

to=time.time()

cations = {b'HIS': (b"HD1", b"HD2", b"HE1", b"HE2"),
        b'ASP':None,
        b"LYS":(b"HZ1", b"HZ2",b"HZ3"),
        b"TYR":(b"HH"),
        b"GLU":None,
        b"CYS":(b'HG'),
        b"ARG": (b"HE11",b"HE12", b"HE21", b"HE22"),
        b"THR":b"HG1",
        b"SER":b"HG",
        b"TRP":b"HE1"}


anions = {b"HIS":(b"ND1", b"ND2"),
          b"ASP":(b"OD1",b"OD2"),
          b"LYS":(b"NZ"),
          b"TYR":(b"OH"),
          b"GLU":(b"OE1", b"OE2"),
          b"CYS":(b"SG"),
          b"ARG": (b"NE1",b"NE2"),
          b"THR":(b"OG1"),
          b"SER":(b"OG"),
          b"TRP":(b"NE1")}

charges = {b'HIS': 1, b'LYS': 1,
           b'ASP': -1, b'GLU': -1,  # carboxylate
           b'CYS': 0,
           b'TYR': 0,
           b"TRP": 1,
           b"ARG": 1,
           b"THR": 0,
           b"SER": 0
           }   # thiol and phenol prot


fullcode={b"HIS":b"0",
        b"ASP":b"1",
        b"LYS":b"2",
        b"TYR":b"3",
        b"GLU":b"4",
        b"CYS":b"5"}
code={b"H":b"0",
        b"A":b"1",
        b"L":b"2",
        b"T":b"3",
        b"G":b"4",
        b"C":b"5"}
elements = {
    b"D":1, b"H": 1, b"C": 6, b"N": 7, b"O": 8, b"S": 16}
    
elements={k.decode() :np.int32(v) for k,v in elements.items()}

class pkparser():
    """#TODO save terminus files
    last id full path include dir. assumed XXXpdb.gz as is gotten from RCSB."""
    """it was necessary to use numpy, because I cant make jagged tensors."""
    def __init__(self,gzipped_pdb):
        self.path = gzipped_pdb#"/Users/jessihoernschemeyer/pKaSchNet/pkas.csv" #('/content/drive/MyDrive/pkas.csv') #3directory
        self.pdb = gzipped_pdb[-7:-3]
        #self.targets=np.load(f"/home/jrhoernschemeyer/Desktop/data_prep/targets/{self.pdb}.npz")
        #self.cutoff=cutoff #Angstrom
        #self.pdbions=[]
        

    def parse_lines_unsupervised(self,lines):

        """input is tlines"""
        
        pdbspecies, pdbcoors,pdbions=[],[],[]
        species,coors,ions,line,lastresi,strii=None,None,[],None,b" ",None
        net=0
        lines=[line.strip(b"abcdefghijklmnopqrstuvwxyz") for line in lines]
        while lines:
            stri=lines[0]
            resi=stri[4:14]
            #print("r",resi)
            #resname=resi[:3]
            #resinum=resi[5:].strip()
            #print(stri,resname,resinum)
                #alt conformations
            
            resi=stri[4:14]
            resname=resi[:3]
            resinum=resi[5:].strip()
            
            if np.char.equal(resinum,lastresi):
                line,lastresi=stri,resinum
                cats=cations[resname]
                if cats:
                    if np.char.startswith(line,cats).any():
                        ions.append(np.int32(1))
                    elif np.char.startswith(line,anions[resname]).any():
                        ions.append(np.int32(-1))
                    else:
                        ions.append(np.int32(0))
                else:
                    if np.char.startswith(line,anions[resname]).any():
                        ions.append(np.int32(-1))
                    else:
                        ions.append(np.int32(0))
                

                species.append(elements[list(filter(str.isalpha, line[-4:].decode()))[0]])
                coors.append((np.float32(line[17:25]),np.float32(line[25:33]),np.float32(line[33:41])))
                    
                del lines[0]
            else: #newresi
                pdbspecies.append(np.array(species))
                pdbcoors.append(np.array(coors))
                net += charges[resname]
                pdbions.append(ions)
                line,lastresi=stri,resinum
                ions=[]
                cats=cations[resname]
                if cats:
                    if np.char.startswith(line,cats).any():
                        ions.append(np.int32(1))
                if np.char.startswith(line,anions[resname]).any():
                    ions.append(np.int32(-1))
                else:
                    ions.append(np.int32(0))
                
                species=[elements[list(filter(str.isalpha, line[-4:].decode()))[0]]]
                coors=[(np.float32(line[17:25]),np.float32(line[25:33]),np.float32(line[33:41]))]
                lastresi=resinum
                del lines[0]

  
                     
        pdbspecies.append(np.array(species))
        pdbcoors.append(np.array(coors))
        pdbions.append(ions)
        self.ions=pdbions[1:]
        self.net = net

        return pdbspecies, pdbcoors
    


    
    def parse_pdb(self):
        """to do try except re: encoding/gzipped"""
        with gzip.open(self.path, "r") as f:
            lines=np.char.array(f.readlines())
            f.close()
        #TODO: encode everything if user didnt gzip their filess

        #run
       
       #get only atom ang hetatm records, split into het and atom
        lines=[line for line in lines if np.char.startswith(line,[b"ATOM"])]
        
        if lines:
            tlines=[line[13:] for line in np.char.array(lines) if line[16:20].strip() in anions.keys()]
            pdbspecies,pdbcoors=self.parse_lines_unsupervised(tlines)
                

        
            return np.array(pdbspecies[1:],dtype=object), np.array(pdbcoors[1:],dtype=object)
        else:
            return 
    


    def run(self):
        """TODO save terminus lines"""


        out=self.parse_pdb()
        #print(len(self.ions[0]),len(out[1][0]),len(out[0][0]))
        if out:
            
            np.savez_compressed(f"/home/jrhoernschemeyer/Desktop/data_prep/egnn_encoder/{self.pdb}.npz",z=out[0], pos=out[1], ions=np.array(self.ions), y=self.net)

        return 
    #except:
    #"zip the pdb or find pdb code somehow else lol or seperate home dir from after"

paths=glob.glob("/home/jrhoernschemeyer/Desktop/data_prep/nometals/zipped-reduced-oddname/*.gz")
for p in paths:
    pkparser(gzipped_pdb=p).run()
    
    

print((time.time()-to)/60, "mins. 6257 pdbs")
#[len(x) for x in np.load("/home/jrhoernschemeyer/Desktop/data_prep/testingunsup_4b00.npz",allow_pickle=True)["z"]]

#8:11. 20mins