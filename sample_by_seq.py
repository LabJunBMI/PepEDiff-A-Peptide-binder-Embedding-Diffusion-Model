import re
import os
import gc
import torch
import pickle
import numpy as np
from torch import nn
from typing import List
from tqdm.auto import tqdm
from transformers import BertConfig
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils import compute_alphas, cosine_beta_schedule
from lightning_fabric.utilities.seed import seed_everything

from model_norm import ConditionalBertForDiffusion
from transformers import T5Tokenizer, T5ForConditionalGeneration

gc.enable()

STEP = 2
GPU_ID = 0
NUM_THREAD = 16
DEVICE = f"cuda:{GPU_ID}"

GEN_NUM = os.environ.get("GEN_NUM", 100)
PEPTIDE_LEN = os.environ.get("PEPTIDE_LEN", 15)

# example of TIGIT
POCKET_IDX = os.environ.get(
    "POCKET_IDX", "45,46,47,48,49,50,51,52,53"
)  # Format: 1,2,3,4,5,...,n
RECEPTOR_SEQ = os.environ.get(
    "RECEPTOR_SEQ",
    "MMTGTIETTGNISAEKGGSIILQCHLSSTTAQVTQVNWEQQDQLLAICNADLGWHISPSFKDRVAPGPGLGLTLQSLTVNDTGEYFCIYHTYPDGTYTGRIFLEVLESSVAEHGARFQIPLLGAMAATLVVICTAVIVVVALTRKKKALRIHSVEGDLRRKSAGQEEWSPSAPSPPGSCVQAEAAPAGLCGEQRGEDCAELHDYFNVLSYRSLGNCSFFTETG",
)


OUTPUT_FILE = os.environ.get("OUTPUT_FILE", "./data/gen_TIGIT_15.pkl")
PROTT5_PATH = "Rostlab/prot_t5_xl_uniref50"
EMBEDDING_SCALER_FILE = "./z_score_scaler.pkl"

MODEL_PATH = "./model.pt"
RANGE_CLIP = (-10, 10)
CONFIG = {
    "pocket_ext": 1,
    "timesteps": 1000,
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


def sample_batch(model, receptor_info):
    # Sample noise and sample the lengths
    ligand_emb_noise = torch.randn_like(receptor_info["receptor_emb"], device=DEVICE)
    sampled = p_sample_loop(
        model=model,
        ligand_emb_noise=ligand_emb_noise,
        ligand_mask=receptor_info["ligand_mask"].to(DEVICE),
        receptor_emb=receptor_info["receptor_emb"].to(DEVICE),
        receptor_mask=receptor_info["receptor_mask"].to(DEVICE),
        pocket_mask=receptor_info["pocket_mask"].to(DEVICE),
        total_timesteps=CONFIG["timesteps"] - 1,
        betas=cosine_beta_schedule(CONFIG["timesteps"]),
        disable_pbar=False,
    )
    # Gets to size (timesteps, seq_len, n_ft)
    ligand_length = [m.sum().int() for m in receptor_info["ligand_mask"]]
    trimmed_sampled = [
        sampled[:, i, :l, :].numpy() for i, l in enumerate(ligand_length)
    ]
    trimmed_sampled = [s[-1] for s in trimmed_sampled]  # extract last time step
    # trimmed_sampled = [(p, s) for p,s in zip(receptor_info["pdb_id"], trimmed_sampled)]
    del ligand_emb_noise, sampled, ligand_length
    return trimmed_sampled


def load_diff_model() -> ConditionalBertForDiffusion:
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


def load_prott5_model():
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(PROTT5_PATH)
    model = T5ForConditionalGeneration.from_pretrained(PROTT5_PATH)
    model = model.eval()
    model = model.to(device)
    return model, tokenizer


def load_emb_scaler():
    with open(EMBEDDING_SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    return scaler


def get_embedding(seq: str, model: T5ForConditionalGeneration, tokenizer: T5Tokenizer):
    seq = re.sub(r"[UZOB]", "X", " ".join(seq))
    # Tokenize the input text
    tokens = tokenizer(seq, add_special_tokens=True, padding=True, return_tensors="pt")
    tokens = tokens.to(DEVICE)
    with torch.no_grad():
        # Pass the input through the encoder
        encoder_outputs = model.encoder(
            input_ids=tokens.input_ids, attention_mask=tokens.attention_mask
        )
        # Extract the hidden states
        encoder_hidden_states = encoder_outputs.last_hidden_state.cpu()
    del tokens, encoder_outputs
    return encoder_hidden_states[0]


def pad(seq, max_len):
    if seq.shape[0] > max_len:
        raise RuntimeError("Length exceed:", len(seq), max_len)
    seq = F.pad(seq, (0, 0, 0, max_len - seq.shape[0]), mode="constant", value=0)
    return seq.float()


def prepare_inputs(
    receptor_emb: torch.Tensor, pocket_idxes: List[int], ligand_len: int
):
    receptor_len = receptor_emb.shape[0]
    ligand_mask = torch.zeros(size=(CONFIG["max_seq_len"],))
    ligand_mask[: ligand_len + 1] = 1.0  # plus one for extra token from embedding
    receptor_mask = torch.zeros(size=(CONFIG["max_seq_len"],))
    receptor_mask[:receptor_len] = 1.0
    pocket_mask = torch.zeros(size=(CONFIG["max_seq_len"],))
    pocket_mask = torch.Tensor(
        [(i in pocket_idxes) for i in range(CONFIG["max_seq_len"])]
    ).bool()
    # pocket_shit_left = torch.roll(pocket_mask, CONFIG["pocket_ext"])
    # pocket_shit_left[0] = False
    # pocket_shit_right = torch.roll(pocket_mask, -CONFIG["pocket_ext"])
    # pocket_shit_right[-1] = False
    # pocket_mask = pocket_mask | pocket_shit_left | pocket_shit_right
    receptor_emb = pad(receptor_emb, CONFIG["max_seq_len"])
    res = {
        "ligand_mask": ligand_mask.repeat(GEN_NUM, 1),
        "receptor_emb": receptor_emb.repeat(GEN_NUM, 1, 1),
        "receptor_mask": receptor_mask.repeat(GEN_NUM, 1),
        "pocket_mask": pocket_mask.repeat(GEN_NUM, 1),
    }
    return res


def has_repeated_AA(s: str, threshold: float = 0.3) -> bool:
    if len(s) <= 3:
        return False
    # Calculate the threshold count
    threshold_count = len(s) * threshold
    # Create a dictionary to count occurrences of each character
    char_count = {}
    # Count occurrences of each character
    for char in s:
        if char in char_count:
            char_count[char] += 1
        else:
            char_count[char] = 1
    # Check if any character exceeds the threshold count
    for count in char_count.values():
        if count > threshold_count:
            return True
    return False


def has_consecutive_AA(s: str, threshold: float = 0.3) -> bool:
    # Calculate the threshold count
    threshold_count = len(s) * threshold
    # Initialize variables to track the current character and its consecutive count
    max_consecutive_count = 0
    current_char = ""
    current_consecutive_count = 0
    # Iterate through the string to count consecutive characters
    for char in s:
        if char == current_char:
            current_consecutive_count += 1
        else:
            current_char = char
            current_consecutive_count = 1
        # Update the max consecutive count
        if current_consecutive_count > max_consecutive_count:
            max_consecutive_count = current_consecutive_count
    # Check if the max consecutive count exceeds the threshold count
    return max_consecutive_count > threshold_count


def emb_to_seq(decoder_model, pred_embedding, scaler):
    all_seqs = []
    for i in pred_embedding:
        emb = torch.Tensor(np.array([scaler.inverse_transform(i)])).to(DEVICE)
        decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids
        decoder_input_ids = decoder_input_ids.to(DEVICE)
        output_ids = decoder_model.generate(
            input_ids=decoder_input_ids, encoder_outputs=(emb,), max_length=emb.shape[1]
        )
        output_seq = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_seq = output_seq.replace(" ", "")
        print(output_seq)
        all_seqs.append(output_seq)
    return all_seqs


def emb_to_seq_noise(decoder_model, pred_embedding, scaler):
    all_seqs = []
    for i in pred_embedding:
        extra_noise_scale = 0.3
        success_flag = False
        gen_round = 0
        emb = torch.Tensor(np.array([scaler.inverse_transform(i)])).to(DEVICE)
        while (not success_flag) and extra_noise_scale != 0:
            if gen_round != 0:
                noised_emb = emb + torch.randn_like(emb) * extra_noise_scale
            else:
                noised_emb = emb
            decoder_input_ids = tokenizer("<pad>", return_tensors="pt").input_ids
            decoder_input_ids = decoder_input_ids.to(DEVICE)
            # Generate the output sequence using the noised hidden states from the encoder
            output_ids = decoder_model.generate(
                input_ids=decoder_input_ids,
                encoder_outputs=(noised_emb,),
                max_length=emb.shape[1],
            )
            output_seq = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            output_seq = output_seq.replace(" ", "")
            success_flag = (not has_consecutive_AA(output_seq)) and (
                not has_repeated_AA(output_seq)
            )
            if (gen_round % 50) == 0 and gen_round != 0:
                extra_noise_scale += 0.1
                print("Increase Noise to:", extra_noise_scale)
            if gen_round >= 500:
                break
            gen_round += 1
        print(tokenizer.decode(output_ids[0], skip_special_tokens=True), gen_round)
        seq = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        all_seqs.append(seq.replace(" ", ""))
    return all_seqs


if __name__ == "__main__":
    torch.set_num_threads(NUM_THREAD)
    seed_everything(CONFIG["random_seed"])
    if len(RECEPTOR_SEQ) == 0:
        raise ValueError(
            "Get Empty Receptor Sequence. Please specifiy by setting the RECEPTOR_SEQ."
        )
    print(
        f"Get receptor sequences:\n{RECEPTOR_SEQ}\n", f"Generating {GEN_NUM} sequences."
    )
    print("Loading ProtT5 Model")
    prott5_model, tokenizer = load_prott5_model()
    print("Calculating Embedding")
    receptor_emb = get_embedding(RECEPTOR_SEQ, prott5_model, tokenizer)
    pocket_idxes = [int(i) for i in POCKET_IDX.split(",")]
    receptor_info = prepare_inputs(receptor_emb, pocket_idxes, PEPTIDE_LEN)
    print("Loading Diffusion Model")
    diff_model = load_diff_model()
    print("Starting Sampling")
    res = sample_batch(diff_model, receptor_info)
    print("Reversing Embdding to Sequence")
    emb_scaler = load_emb_scaler()
    all_seqs = emb_to_seq_noise(prott5_model, res, emb_scaler)
    with open(OUTPUT_FILE, "+wb") as f:
        pickle.dump(all_seqs, f)
