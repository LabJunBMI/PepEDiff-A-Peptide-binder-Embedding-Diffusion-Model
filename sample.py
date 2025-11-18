import gc
import torch
import pickle
from torch import nn
from tqdm.auto import tqdm
from utils import compute_alphas
from transformers import BertConfig
from torch.utils.data import DataLoader
from lightning_fabric.utilities.seed import seed_everything

from dataset import LigandBindingSiteDataset
from model_norm import ConditionalBertForDiffusion

gc.enable()

STEP = 2
GPU_ID = 0
NUM_THREAD = 16
DEVICE = f"cuda:{GPU_ID}"
OUTPUT = "./data/{batch_idx}_{random_seed}.pkl"
MODEL_PATH = "./model.pt"
DATA_FILE = "./seq_emb_1024_no_x_z_test.pkl"
CONFIG = {
    "pocket_ext": 1,
    "timesteps": 999,
    "max_seq_len": 1024,
    "num_heads": 8,
    "dropout_p": 0.1,
    "hidden_size": 2048,
    "num_hidden_layers": 2,
    "intermediate_size": 4096,
    "position_embedding_type": "relative_key",
    "lr": 5e-5,
    "l2_norm": 0.1,
    "loss": "smooth_l1",
    "gradient_clip": 1.0,
    "lr_scheduler": "LinearWarmup",
    "min_epochs": 40,
    "max_epochs": 500,
    "batch_size": 8,
    "random_seed": 0,
    "feature_size": 1024,
}


@torch.no_grad()
def p_sample(
    model: ConditionalBertForDiffusion,
    ligand_emb_noise: torch.Tensor,
    ligand_mask: torch.Tensor,
    receptor_emb: torch.Tensor,
    receptor_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    timestep: int,
    betas: torch.Tensor,
) -> torch.Tensor:
    if timestep <= 1:  # skip timestep=0
        return ligand_emb_noise, ligand_emb_noise
    b = ligand_emb_noise.shape[0]
    alpha_beta_values = compute_alphas(betas)
    sqrt_recip_alphas = 1.0 / torch.sqrt(alpha_beta_values["alphas"])
    sqrt_recip_alphas_t = sqrt_recip_alphas[timestep]
    betas_t = betas[timestep]
    sqrt_one_minus_alphas_cumprod_t = alpha_beta_values[
        "sqrt_one_minus_alphas_cumprod"
    ][timestep]
    noise_pred = model(
        torch.full((b,), timestep, device=DEVICE, dtype=torch.long),
        ligand_emb_noise,
        ligand_mask,
        receptor_emb,
        receptor_mask,
        pocket_mask,
    )
    model_mean = sqrt_recip_alphas_t * (
        ligand_emb_noise - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = alpha_beta_values["posterior_variance"][timestep]
    noise = torch.randn_like(ligand_emb_noise)
    return model_mean + torch.sqrt(posterior_variance_t) * noise, noise_pred


@torch.no_grad()
def p_sample_loop(
    model: nn.Module,
    ligand_emb_noise: torch.Tensor,
    ligand_mask: torch.Tensor,
    receptor_emb: torch.Tensor,
    receptor_mask: torch.Tensor,
    pocket_mask: torch.Tensor,
    total_timesteps: int,
    betas: torch.Tensor,
    disable_pbar: bool = False,
) -> torch.Tensor:
    """
    Returns a tensor of shape (timesteps, batch_size, seq_len, n_ft)
    """
    print("init noise:", ligand_emb_noise.max(), ligand_emb_noise.min())
    b = ligand_emb_noise.shape[0]
    noises = []
    tqdm_bar = tqdm(
        reversed(range(0, total_timesteps, STEP)),
        desc="sampling loop time step",
        total=int(total_timesteps / STEP),
        disable=disable_pbar,
    )
    for i in tqdm_bar:
        # Shape is (batch, seq_len, 1024)
        ligand_emb_noise, pred_noise = p_sample(
            model=model,
            ligand_emb_noise=ligand_emb_noise,
            ligand_mask=ligand_mask,
            receptor_emb=receptor_emb,
            receptor_mask=receptor_mask,
            pocket_mask=pocket_mask,
            timestep=i,
            betas=betas,
        )
        tqdm_bar.set_postfix(
            emb_min=float(ligand_emb_noise.min()),
            emb_max=float(ligand_emb_noise.max()),
            pred_min=float(pred_noise.min()),
            pred_max=float(pred_noise.max()),
        )
        noises.append(ligand_emb_noise.cpu())
    del b, ligand_emb_noise
    return torch.stack(noises)


def sample_batch(model, test_ds: LigandBindingSiteDataset):
    test_dataloader = DataLoader(
        test_ds, batch_size=CONFIG["batch_size"], prefetch_factor=2, num_workers=16
    )
    for batch_idx, batch in enumerate(test_dataloader):
        print(f"Generating Batch {batch_idx}/{len(test_dataloader)}")
        # Sample noise and sample the lengths
        ligand_emb_noise = torch.randn_like(batch["ligand_emb"], device=DEVICE)
        sampled = p_sample_loop(
            model=model,
            ligand_emb_noise=ligand_emb_noise,
            ligand_mask=batch["ligand_mask"].to(DEVICE),
            receptor_emb=batch["receptor_emb"].to(DEVICE),
            receptor_mask=batch["receptor_mask"].to(DEVICE),
            pocket_mask=batch["pocket_mask"].to(DEVICE),
            total_timesteps=CONFIG["timesteps"],
            betas=test_ds.alpha_beta_terms["betas"],
            disable_pbar=False,
        )
        # Gets to size (timesteps, seq_len, n_ft)
        ligand_length = [m.sum().int() for m in batch["ligand_mask"]]
        trimmed_sampled = [
            sampled[:, i, :l, :].numpy() for i, l in enumerate(ligand_length)
        ]
        trimmed_sampled = [s[-1] for s in trimmed_sampled]  # extract last time step
        trimmed_sampled = [(p, s) for p, s in zip(batch["pdb_id"], trimmed_sampled)]
        with open(
            OUTPUT.format(batch_idx=batch_idx, random_seed=CONFIG["random_seed"]), "+wb"
        ) as f:
            pickle.dump(trimmed_sampled, f)
        del ligand_emb_noise, sampled, ligand_length, trimmed_sampled
    return None


def get_test_dataset(file_path):
    test_ds = LigandBindingSiteDataset(file_path)
    return test_ds


def load_model():
    bert_config = BertConfig(
        max_position_embeddings=CONFIG["max_seq_len"],
        num_attention_heads=CONFIG["num_heads"],
        hidden_size=CONFIG["hidden_size"],
        intermediate_size=CONFIG["intermediate_size"],
        num_hidden_layers=CONFIG["num_hidden_layers"],
        position_embedding_type=CONFIG["position_embedding_type"],
        hidden_dropout_prob=CONFIG["dropout_p"],
        attention_probs_dropout_prob=CONFIG["dropout_p"],
        use_cache=False,
    )
    model = ConditionalBertForDiffusion(
        feature_size=CONFIG["feature_size"],
        lr_scheduler=CONFIG["lr_scheduler"],
        epochs=CONFIG["max_epochs"],
        l2_lambda=CONFIG["l2_norm"],
        learning_rate=CONFIG["lr"],
        bert_config=bert_config,
    )
    model.load_state_dict(torch.load(MODEL_PATH))
    model = model.eval().to(DEVICE)
    return model


if __name__ == "__main__":
    # torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(NUM_THREAD)
    test_dataset = get_test_dataset(DATA_FILE)
    model = load_model()
    for i in range(10):
        CONFIG["random_seed"] = i
        seed_everything(CONFIG["random_seed"])
        sample_batch(model, test_dataset)

    # sample_result = sample_batch(model, test_dataset)
    # with open(OUTPUT, "+wb") as f:
    #     pickle.dump(sample_result, f)