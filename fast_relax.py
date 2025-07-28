import pyrosetta
from pyrosetta import pose_from_pdb, get_score_function
from pyrosetta.rosetta.protocols.relax import FastRelax

pyrosetta.init()

import multiprocessing as mp
import time
import os

COMBINED_FOLDER = "/home/liangpu/drug_discovery_data/code/simple_seq_emb_binder_gen/combined/seq_checked_7"
RELAXED_FOLDER = "/home/liangpu/drug_discovery_data/code/simple_seq_emb_binder_gen/relaxed/seq_checked_7"

def fast_relax(pdb_path, output_path):
    # Try to avoid unknow multi-thread errors
    scorefxn = get_score_function()
    relax = FastRelax()
    relax.set_scorefxn(scorefxn)

    pose = pose_from_pdb(pdb_path)
    relax.apply(pose)
    pose.dump_pdb(output_path)

def fast_relax_wrapper(structure_id):
    pdb_path = os.path.join(COMBINED_FOLDER, f"{structure_id}.pdb")
    output_path = os.path.join(RELAXED_FOLDER, f"{structure_id}.pdb")
    fast_relax(pdb_path, output_path)

if __name__ == "__main__":
    relaxed_ids = [s.split(".")[0] for s in os.listdir(RELAXED_FOLDER)]
    structure_ids = [s.split(".")[0] for s in os.listdir(COMBINED_FOLDER)]
    unrelaxed_ids = [i for i in structure_ids if i not in relaxed_ids]
    print(len(unrelaxed_ids))
    time.sleep(10)
    with mp.Pool(12) as pool:
        result = pool.map(fast_relax_wrapper, unrelaxed_ids)