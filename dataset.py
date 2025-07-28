import torch
import pickle
import random
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from typing import List, Optional
from torch.utils.data import Dataset, DataLoader
from utils import cosine_beta_schedule, compute_alphas

RANDOM_SEED = 0

class LigandBindingSiteDataset(Dataset):
    def __init__(
        self, filepath: str, split_name: str = None, split_ids: List[str] = None, 
        max_len: int = 1024, pocket_ext: int = 1, timesteps: int = 1000
    ) -> None:
        """Create dataset
        Args:
            split (str):
                String of train, validation
            filepath (str):
                A filepath of pickle file of dataframe with columns:
                    ligand_angle,binding_site_sequence
            tokenizer (PreTrainedTokenizer):
                A tokenizer for proteins
            ligand_min_len (int):
                minmun length for sequences
            max_len (int):
                Maximun length for sequences
        """
        super().__init__()
        self.max_len = max_len
        self.pocket_ext = pocket_ext
        self.timesteps=timesteps
        self._load_file(filepath)
        self._split_data(split_ids, split_name)
        betas = cosine_beta_schedule(timesteps)
        self.alpha_beta_terms = compute_alphas(betas)

    def _split_data(self, split_ids=None, split_name=None):
        print("Spliting Data")
        if(split_name!=None and split_ids!=None):
            target_split_ids = split_ids[split_name]
            self.data = [d for d in self.data if d["pdb_id"] in target_split_ids]

    def _load_file(self, filepath: str) -> None:
        print(f"Loading data from {filepath}")
        with open(filepath, "rb") as f:
            self.data = pickle.load(f)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index, use_timestep: Optional[int] = None,):
        """
        Dict[
            'receptor_emb', 
            'receptor_mask', 
            'ligand_emb', 
            'ligand_mask', 
            'pocket_mask', 
            'pdb_id', 
            'timestep', 
            'sqrt_alphas_cumprod_t', 
            'sqrt_one_minus_alphas_cumprod_t', 
            'noise', 
            'noised_ligand_emb'
        ]
        """
        if use_timestep is not None:
            timestep = np.clip(np.array([use_timestep]), 0, self.timesteps - 1)
            timestep = torch.from_numpy(timestep).long()
        else:
            timestep = torch.randint(0, self.timesteps, (1,)).long()
        if not 0 <= index < len(self):
            raise IndexError("Index out of range")
        item:dict = self.data[index]
        item.update(self._get_alpha_term(timestep))
        noised_values = self._get_noise_by_timestep(
            item["ligand_emb"],
            item["sqrt_alphas_cumprod_t"],
            item["sqrt_one_minus_alphas_cumprod_t"],
        )
        item.update({
            "noise":noised_values["noise"],
            "noised_ligand_emb":noised_values["noised_value"]
        })
        return item

    def _get_noise_by_timestep(
        self, v: torch.Tensor, sqrt_alphas_cumprod_t, sqrt_one_minus_alphas_cumprod_t
    ):
        noise = torch.randn_like(v)  # Vals passed in only for shape
        noised_value = (
            sqrt_alphas_cumprod_t * v + sqrt_one_minus_alphas_cumprod_t * noise
        )
        return {
            "noise": noise,
            "noised_value": noised_value,
        }

    def _get_alpha_term(self, timestep: torch.Tensor):
        sqrt_alphas_cumprod_t = self.alpha_beta_terms["sqrt_alphas_cumprod"][
            timestep.item()
        ]
        sqrt_one_minus_alphas_cumprod_t = self.alpha_beta_terms[
            "sqrt_one_minus_alphas_cumprod"
        ][timestep.item()]
        return {
            "timestep": timestep,
            "sqrt_alphas_cumprod_t": sqrt_alphas_cumprod_t,
            "sqrt_one_minus_alphas_cumprod_t": sqrt_one_minus_alphas_cumprod_t,
        }
"""

from dataset import *
ds = LigandBindingSiteDataset("/data/bai/Drug_discovery/cleanData/BioLip/seq_emb_1024_z_test.pkl")

dl = DataLoader(ds, 4)
for i in dl:
    break
"""