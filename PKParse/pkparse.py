import numpy as np
import gzip
import glob
import edlib
import os
from utils import constants


#data_dir = os.getcwd()
constants = constants()
fullcode = constants["titratables_full"]
code = constants["titratables_short"]
three_to_one = constants["aa_code"]
elements = constants["elements"]
cof = constants["cofactors"]
ligands = constants["ligands"]
#cnbrs = constants["disulfide_nn"]
amber=constants["amber_sites"]



class parser():
    def __init__(self, gzipped_pdb, ref_pdb):
        self.path    = gzipped_pdb
        self.pdb     = gzipped_pdb[-7:-3]
        self.targets =f"../../data/targets/{self.pdb}.npz"
        self.save_dir = "../inputs/"
        self.ligands = ligands
        self.ref_pdb = ref_pdb
        
    def residue_map(self):
            """This function uses the RCSB reference pdb as the reference sequence
            for sequence alignment through edlib. This allows for the incoming residue 
            numbers to be arbitrary and allows for custom PDDs and to be fixed by modeling
            software (does not support files with non-standardized naming conventions, where
            standard is considered to be Amber, even if that isn't correct.)

            **does not support files without:
                 TER records 
                 Chain records ("A", "X")
                 gzipped

            cur = modeled and parsed | ref = rcsb (assumed pkPDB numbering)

            three_to_one is a dict takes as input key the residue name, which if it is standard as it should be, converts it to its one letter nickname e.g. GLU --> E. 
            If it is not recognized, e.g. a typesetter error in original PDB making GLX for GLU, the nickname will be "X". Currently, these will still be paired as
            two X's rather than excluded on the basis of being an X. However, if this code is used as intended with a PDBFixed PDB or analogous (find nonstandard residue
            names and replace with standard names built in), this should ideally not occur where two Xs are paired. In any case, it will not affect our results if two
            Xs are paired, because the Xs will not ever be our five titratable residues which needed the standard names "HIS/ASP/etc." to be parsed. The fact that 
            every residue is used in this function is only because this is needed for full integrity alignment. 

            The output is matched pairwise such that insertions and deletions trigger 
            the increase of their respective counter, which retrieves their keys made by
            residue number, chain, and residue name e.g. 12A0 = residue 12, chain A, HIS 
            (where HIS = 0 is an internal dictionary code defined in utils.py.)

            These matched residues form the mapping which are the only IDs which get through
            the parser to be matched with the pkPDB targets.

            ##New##
            Manual parsing to retrieve sequence: Atom-ine by Atom-line (exclude ligands) the code retrieves the protein sequence.
            Residue by residue, the IDs are formed (12A0 = res12,chA,HIS).
            Chain by chain (triggered by "TER"), the chain-sequences are made (b"EEH" = GLU GLU HIS) and appended to protein-wide chain list "seqs",
            as well as their corresponding IDs ([b"1A4",b"2A4",b"3A0"] == Glu Glu His) 
            During parse, the last residue is remembered and inaction taken upon its subsequent parsing.

            Once the sequences and key lists are formed, they are input into edlib. 
            The keys will be parsed by index and thus doesnt include the deletions which will appear as "-"'s in edlibs nice alignment.

            After the alignment is made, the matches are paired such that the index value slicing the og/model IDs list         #TODO: rename for clarity?
            increases when there is a match and when there isnt. When there is, a key value pair is made between the             #TODO: save mapping? woulda been good..
            residue IDs ("1A4" <-->"100A4") between the OG residue and the modeled, thereby mapping any two residue numbering schemes.
            When deletion in the modeled (should not occur, ever, for this project since at the baseline "modeled" PDBs are the RCSB itself, but 
            121k PDBs = mysteries), it should mean the next residue in the references matches with the current frame of the nidx, which wouldnt have
            received a number during enumeration as it enumerated on the unaligned sequences and thus no "-"'s. This is why i, despite it retrieves
            from reference seq, is incremented upon deletion in the modeled sequence.

            The opposite logic is true for additions in the modeled structure with missing residues in the original strucute inducing an increase in
            the index slicing the modeled pdb's residue ID keys. This is assumed to be the baseline behavior for this project

    

            """
            # helper: extract per-chain sequences & idx-lists into parallel lists
            def extract(path):
                seqs,seq,idxs,keys = [],[],[],[]   # list of lists of idx‐bytes
                lastkey=None
                with gzip.open(path, "rb") as fh:
                    for raw in fh:
                        if not raw.startswith(b"ATOM"): continue
                        elif raw.startswith(b"TER"):
                            seqs.append("".join(seq))
                            idxs.append(keys)
                            keys,seq=[],[]
                            continue

                        ch,resi = raw[21:22], raw[17:20]
                        key = raw[22:26].strip() + ch + fullcode.get(resi,b"") #code is {"HIS":0..}
                        
                        if lastkey==key:
                            lastkey=key
                            continue

                        keys.append(key)
                        aa1 = three_to_one.get(resi, "X") 
                        seq.append(aa1)
                        lastkey=key
                        
                seqs.append("".join(seq))
                idxs.append(keys)

                # finalize: join seqs, convert idxs to np.char.array
                seq_strs = seqs#["".join(s) for s in seqs]
                idx_arrs  = [np.char.array(i) for i in idxs]
                return seq_strs, idx_arrs

            # read ref & current
            ref_seqs, ref_idxs = extract(self.ref_pdb)
            cur_seqs, cur_idxs = extract(self.path)

            mapping = {}
            # align each chain by index
            for c, oseq in enumerate(ref_seqs):
                if c >= len(cur_seqs):
                    break
                nseq  = cur_seqs[c]
                nidx = cur_idxs[c]
                oidx = ref_idxs[c]

                # 1) compute the alignment path
                res = edlib.align(oseq, nseq, mode="NW", task="path")
                # skip if no CIGAR produced
                if not res.get("cigar"):
                    # you could log a warning here if you like:
                    print(f"Warning: no alignment path for chain {c}, skipping")
                    self.mapping=None
                    return

                nice = edlib.getNiceAlignment(res, oseq, nseq)
                ref_aln, qry_aln = nice["target_aligned"], nice["query_aligned"]
                i = j = 0
                for r_c, q_c in zip(ref_aln, qry_aln):
                    if r_c != "-":
                        if q_c != "-":
                            if r_c == q_c:
                                mapping[nidx[j]] = oidx[i]
                                i += 1; j += 1
                            else: 
                                i += 1; j += 1
                        else: j += 1 #insertion
                    else: i += 1 #deletion

            self.mapping = mapping

    def parse_titratable_lines(self, lines):
        """get info from asp,glu,his,cys,tyr. removes hydrogens. 
        
        It strips residue numbers of their insertions, which have been fixed by a mender which never has two residue numbers the same. The chance of 
        failure is probably small but still real if this is used not as intended e.g. PARSING A PDB WHICH HAS INSERTION CODES THAT DONT BY
        A GOOD CHANCE HAVE A UNIQUE RES NUMBER EXCLUDING INSERTION CODE (A LETTER) FOR EACH RESIDUE IN A CHAIN.

        
        If there is a titratable heavy atom that is the hydrogen pair doner/acceptor defined by Amber and thus in amber_set, then the residue-wise information is 
        retained in the species/coors long term memory. If the parser never encountered a titratable site (e.g. parsing a raw rcsb pdb with unresolved Lysine side chain
        and thus a missing NZ atom), then the residue information will instead be sent to "others", which is the atoms from non-titratable including
        ligands, residues, and non-labeled titratable residues."""
            
        amber_set = amber.values()      # byte-strings of titratable names
        self.species,self.coors, self.sites  = [],[],[]
        last_resnum                                   = None
        cur_species, cur_coords     =[],[]
        amber_flag=False
        cur_species, cur_coords, others_c,others_s      =[],[],[],[]

        # 1) Single-pass parse & group by residue
        for line in lines:
            try:
                #skip hydrogens
                if line.lstrip().startswith(b"H"): continue
                #strip insertions 
                resnum = line[10:15].strip(b"ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            

                # new residue?  put previous residue in long term memory of potential sites (self.species), or other resis (self.others) depending on flag status
                if resnum != last_resnum and cur_species:
                    if amber_flag: 
                        self.species.append(cur_species)
                        self.coors.append(cur_coords)
                    else: 
                        others_s.append(cur_species)
                        others_c.append((cur_coords))
                    cur_species, cur_coords = [], []
                    amber_flag=False
                
                #per-residue short term memory accumulates Z + pos info
                cur_species.append(elements[line[-9:].split()[0][0:]])
                cur_coords.append((float(line[18:26]), float(line[26:34]), float(line[34:42])))

                #only retain resis with resolved/modeled titratable sites. these lines generate the IDs to be paired with pkPDB targets downstream
                if line[:5].strip() in amber_set:
                    amber_flag=True
                    self.sites.append(line)

                last_resnum = resnum
            except Exception as e:
                print(self.pdb)
                print(e)
                print(line)
                return False
                

            
        #final long term memory capture
        if amber_flag: 
            self.species.append(cur_species)
            self.coors.append(cur_coords)
        else: 
            others_s.append(cur_species)
            others_c.append((cur_coords))
        if others_s: self.others.append((np.concatenate(others_s),np.vstack(others_c)))

        #save as arrays for downstream mask operations
        self.species,self.coors=np.array(self.species,dtype=object),np.array(self.coors,dtype=object)
        return True 
        
    
    def aggregate_others(self):
        """#TODO"""
        #def stack_columns(data):
        """

        It simply reshapes all the information ive thrown into self.others until it is called in run().

        ###############
        data: list of (int_arr, coord_arr) tuples, where
        - int_arr is either a 1D np.ndarray of ints or an object array of int sub-arrays
        - coord_arr is either a 2D np.ndarray of shape (M,3) or an object array of 2D sub-arrays
        Returns:
        - all_ints: 1D np.ndarray of all ints concatenated
        - all_coords: 2D np.ndarray of shape (total_rows, 3)
        """
        int_chunks, coord_chunks = [], []
        for int_arr, coord_arr in self.others:
            int_chunks.append(int_arr)
            coord_chunks.append(coord_arr)
        all_ints   = np.concatenate(int_chunks, axis=0)
        all_coords = np.vstack(coord_chunks)
        return all_ints, all_coords

    
    def parse_others(self, lines):
        """parse all other ligand HETATM and non-ligand ATOM lines using entire periodic table"""
        species, coors = [], []
        for line in lines:
            # skip hydrogens
            if not line[0:2].strip().startswith(b"H"):
                #here I use cofactors as my Z number dict --> entire periodic table
                species.append(cof[line[-9:].split()[0][0:]])
                coors.append((float(line[18:26]), float(line[26:34]), float(line[34:42])))
                
        self.others.append((species, coors))

    def parse_pdb(self):
        """to do try except re: encoding/gzipped
        #hi b'HG23 ILE A 492      65.222 102.163  26.506  1.00  0.00           H   std\n'
        # #encode everything if user didnt gzip their filess? #TODO
        # TODO: here is where we can encode user input.

        Ligands excludes solvent molecules (exlclusion by lack of membership in ligands).
        Should a solvent molecule have a residue name of a metal of another element,
        they will be parsed!
    

        This code opens the gzipped pdb and line by line extracts ATOM and HETATM records.
        
        For atom records, if they are titratable, they are sent to parse_titratable_lines.
        This uses a Z table dict which is only 6,7,8,16 (C,N,O,S).

        Otherwise, they are sent to parse_others, which parses nontitratable residues (ATOM),
        metal (or non-metal pure elements, but goal is metal) cofactors (HETATM records),
        and ligands (HETATM) with the full periodic table.*

        Others is continuously appended even outside (after) this function so nobody gets left behind.
        
        *Despite there is H in the cofactors dict, only lines which dont have Hydrogens 
        (see parse_others) enter the dictionary."""

        with gzip.open(self.path, "rb") as f: #TODO?
            lines=f.readlines()

        titratables, others  = [], []

        #get ligands
        for line in lines:
            if line.startswith(b"HETATM"):
                if line[16:20].strip() in self.ligands or cof: others.append(line[12:])
            elif line.startswith(b"ATOM"):
                if line[16:20].strip() in fullcode: titratables.append(line[12:])
                else: others.append(line[12:])

        self.parse_others(others)
        flag = self.parse_titratable_lines(titratables) #skips lines without an element in my dictionaries. e.g. Deterium, ionized O and N.
        return flag
        
    def get_disulfides(self,lines):
        """in: the SITE of the titratable lines and thus Sulfur when Cys
        intend to minic pdb2pqr 2.05 cutoff was my understanding
    
        for cys sulfur lines the coords of the sulfurs are entered into a brute force KNN with radius 2.1 (room from 2.05 for error).
        If they have other sulfur neighbors, who are the neighbors are retained and they are sent to others and their IDs get removed
         with masks.
          
           Note it doesnt need to work (=correctly identify) with our labeled data thus was for development and to not forget until unlabeled disaster
           
           somebodz could test if it worked by running same code run on jesses computer with it hashed out and seeing that the lengths of the pk arrays
           is the same and if not, by how much."""
        cys_lines = [(i,line) for i,line in enumerate(lines) if line[5:6] == b"C"]
        if cys_lines:   
            cys_coors = [(float(line[1][18:26]),float(line[1][26:34]),float(line[1][34:42])) for line in cys_lines] #TODO confirm its S?
            cnbrs.fit(cys_coors)
            nbrs = cnbrs.radius_neighbors(cys_coors,radius=2.1, return_distance=False) #TODO
            bridges=[]
            for a,cys in zip(nbrs,cys_lines):
                if len(a) == 2:
                    bridges.append(cys[0])
            self.disulfides=bridges
        else: self.disulfides=None
        
    def run(self):
        """it gets the residue map between the RCSB and assumed pkPDB resi numbering scheme. using edlib seq alignment and parsing 2 pdbs.
        
        Then, the pdb is parsed line by line and hydrogens stripped. Here, the sites corresponding to S, NZ, etc., are gathered. If no
        site, it is sent to others instead of species. that is flag based

        The ids for the modeled and thus parsed PDb are gotten directly from the titratable site ATOM lines e.g. S, NZ..
        if they exist also in the RCSB pdb, they are assumed a potential candidate for having a pypka label and retained
        in species/coors and made an ID. Here I form the sitecoors, too, are the centers for forming neighborhoods 
        downstream.

        I append retained sites (being resolved in OG PDB) for bookkeeping 

        Then, the cys bridges determined with KNN brute force are actually never removed from species. 
        I suppose I did this on purpose as I didnt wanna
        miss no labels developing, but would fail unlabeled data #TODO #ALERTA

        TODO: remove cysteines from SPECIES/COORDS/ (or? ) IDS, or remove disulfides because it is redundant sending
        unmatched labels which include cys  bridges, to others, after already sending them there after disulfides


        final mask:
        Only the structure info from modeled residues is retained. We return the indices corresponding to the intersection
        of the IDs which for the modeled pdb, should be the same length as the species/coords by design but not default #TODO easy integrity check

        and most of the time is the same as pkPDB targets/sites but far from always, especially with modeled missing residued
        in modeled parsed PDB.

        The official IDs are those which are shared by both, and thus not my PDBs. lol. forgot to save map until past execute. good thing code is reproducible
    

        
        """
        
        ""
        
        ""
        if not os.path.exists(self.targets): #skip target files with only ntr and ctr, that I alreadz deleted
            return
            
        
        #get species/pos info of titratable residues
        self.residue_map()
        if not self.mapping:
            with gzip.open("../badparse.gz", "ab") as f:
                f.write(self.pdb.encode())
                f.write(b"\n")
                return
        else:
            #print(self.mapping.items())
            #with gzip.open("../res_maps.gz", "ab") as f:
            #    f.write(b"\n")
            #    f.write(self.pdb.encode())
            #mapping=tuple(self.mapping.items())
            #print(mapping)
            #new, old = list(mapping[0]),list(mapping[1])
            #print(new,old)
            #np.save(np.array)
            map=list(self.mapping.items())
            np.savez_compressed(f"../res_maps/{self.pdb}.npz", map=map)

                

          
        


        
        self.others=[]
        sitecoors,ids,sites=[],[],[]
        
        flag=self.parse_pdb()
        if not flag: #if there was an issue in parsing the elements. should produce errors for Deterium, N1+, O-, and EP.
            with gzip.open("../badparse.gz", "ab") as f:
                f.write(self.pdb.encode())
                f.write(b"\n")
                return
    
        
        for line in self.sites:
            id=line[10:18].strip() + line[9:10] + code[line[5:6]] #get ids frrom sites
            id=self.mapping.get(id) #from map
            if id:
                ids.append(id)
                sitecoors.append((float(line[18:26]),float(line[26:34]),float(line[34:42])))
                sites.append(line)
        sitecoors=np.array(sitecoors).astype(np.float32)
       #that fixes when there is not a titratable site in the titratable residue. now onto targets
        
        #self.get_disulfides(sites) 
        #if self.disulfides: 
        #    sulf=np.array(self.disulfides)
        #    self.others.append((np.concatenate(self.species[sulf]), np.vstack(self.coors[sulf]))) #ALERTA

        pkpdb=np.load(self.targets)
        common, pidx, midx = np.intersect1d(pkpdb["ids"], ids,
                                            return_indices=True)
        site_mask = np.ones(len(self.species), dtype=bool)
        site_mask[midx] = False                            # False → keep

        if site_mask.any():                                # spill invalid sites
            self.others.append((np.concatenate(self.species[site_mask]),
                                np.vstack(self.coors[site_mask])))
            
        self.coors,self.species = self.coors[midx], self.species[midx]
        others=self.aggregate_others()
        #self.hoods(self.coors[midx], self.species[midx])#, self.sitecoors[midx])
        atom_coords = np.concatenate([*self.coors, np.vstack(others[1])], axis=0).astype(np.float32)
        atom_elems  = np.concatenate([*self.species,
                                      others[0]]).astype(np.uint8)


        np.savez_compressed(
    self.save_dir + f"{self.pdb}.npz",
    z=atom_elems,
    pos=atom_coords,
    pks=pkpdb["pks"][pidx],
    ids=common,
    sites=sitecoors[midx]

)
        return self

#to do rerun with right cysteins and keep map this time