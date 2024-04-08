'''
generating PDB files for proteins in the dataset according to atom coordinates and residue types
'''
import os,numpy as np
from tqdm import tqdm
from Bio.PDB import Structure,Model,Residue,Atom,PDBIO,Chain
from Bio.PDB.Polypeptide import PPBuilder

atom_bet = {'N':'N','CA':'C','C':'C','O':'O'}

def create_pdb(residues,name):
    # 创建一个PDB结构
    structure = Structure.Structure(name.split('.pdb')[0])

    # 创建一个模型
    model = Model.Model(0)
    structure.add(model)

    chain = Chain.Chain(residues[0]['chain_id'])
    model.add(chain)
    # 添加残基
    for residue_info in residues:
        res_name = residue_info['res_name']
        chain_id = residue_info['chain_id']
        res_seq = residue_info['res_seq']
        atoms = residue_info['atoms']

        # 创建残基对象
        residue = Residue.Residue((' ', res_seq, ' '), res_name, res_seq)

        # 添加原子
        for atom_name, atom_coord in atoms.items():
            atom = Atom.Atom(atom_name, atom_coord, 0, 1, ' ', atom_name, 0, atom_bet[atom_name])
            residue.add(atom)

        # 将残基添加到模型中
        chain.add(residue)

    # 将PDB写入文件
    io = PDBIO()
    io.set_structure(structure)
    io.save('/media/ST-18T/Ma/ProteinDesign/my/M-Design/data/pdb_files/pdb_files_reconstruction1/'+name)

# alphabet = {'ALA':'A','CYS':'C','ASP':'D','GLU':'E','PHE':'F','GLY':'G','HIS':'H','ILE':'I','LYS':'K','LEU':'L','MET':'M','ASN':'N','PRO':'P','GLN':'Q','ARG':'R','SER':'S','THR':'T','VAL':'V','TRP':'W','TYR':'Y'}

alphabet = {'A':'ALA','C':'CYS','D':'ASP','E':'GLU','F':'PHE','G':'GLY','H':'HIS','I':'ILE','K':'LYS','L':'LEU','M':'MET','N':'ASN','P':'PRO','Q':'GLN','R':'ARG','S':'SER','T':'THR','V':'VAL','W':'TRP','Y':'TYR'}

seq_dict = {}
with open('/media/ST-18T/Ma/ProteinDesign/my/M-Design/data/Sequence.txt') as f:
    for line in f:
        id = line.strip().split('\t')[0]
        seq = line.strip().split('\t')[1]
        seq_dict[id] = seq

path = '/media/ST-18T/Ma/ProteinDesign/my/M-Design/data/Coordinates/'
coord_files = os.listdir(path)
for i in tqdm(coord_files):
    id = i.split('.npy')[0]
    if '.' in id:
        chain_id = id.split('.')[1]
    elif '_' in id:
        chain_id = id.split('_')[1]
    elif len(id) == 4:
        chain_id = 'A'
    seq = [alphabet[aa] for aa in seq_dict[id]]

    coord = np.load(path+i)

    N_coord=coord[0,:,0,:]
    CA_coord=coord[0,:,1,:]
    C_coord=coord[0,:,2,:]
    O_coord=coord[0,:,3,:]

    index = 1

    residues = []

    for idx in range(len(seq)):
        residue = {
            'res_name': seq[idx],
            'chain_id': chain_id,
            'res_seq': idx+1,
            'atoms': {
                'N': tuple(N_coord[idx,:]),
                'CA': tuple(CA_coord[idx,:]),
                'C': tuple(C_coord[idx,:]),
                'O': tuple(O_coord[idx,:])
            }
        }
        residues.append(residue)
    create_pdb(residues,id+'.pdb')
