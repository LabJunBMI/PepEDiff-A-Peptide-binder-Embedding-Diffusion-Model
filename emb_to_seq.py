from transformers import T5Tokenizer, T5ForConditionalGeneration
from tqdm import tqdm
import numpy as np
import pickle
import torch
import gc

gc.enable()
torch.manual_seed(0)
torch.set_num_threads(16)
DEVICE = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
EMB_FILE = "/home/liangpu/simple_seq_emb_binder_gen/data/sample_no_x_res_loss/{batch_idx}_{seed}.pkl"
SEQ_FILE = "/home/liangpu/simple_seq_emb_binder_gen/data/sample_no_x_res_loss/seq_checked_{seed}.pkl"
SCALER_FILE = "/home/liangpu/simple_seq_emb_binder_gen/z_score_scaler.pkl"

def load_all_embedding():
    print("Loading Embedding Files")
    res = {}
    for seed in range(10):
        all_pred_emb = []
        for batch_idx in range(39):
            file_path = EMB_FILE.format(batch_idx=batch_idx, seed=seed)
            with open(file_path, "rb") as f:
                all_pred_emb.extend(pickle.load(f))
            # print(file_path)
        res[seed] = all_pred_emb
    return res

def load_embedding_scaler():
    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)
    return scaler

def load_model():
    tokenizer = T5Tokenizer.from_pretrained('../ProtT5/prot_t5_xl_uniref50', local_files_only=True)
    model = T5ForConditionalGeneration.from_pretrained('../ProtT5/prot_t5_xl_uniref50', local_files_only=True)
    model = model.eval().to(DEVICE)
    return model, tokenizer

def has_repeated_AA(s: str, threshold: float=0.3) -> bool:
    if(len(s)<=3):
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

def has_consecutive_AA(s: str, threshold: float=0.3) -> bool:
    if(len(s)<=3):
        return False
    # Calculate the threshold count
    threshold_count = len(s) * threshold
    # Initialize variables to track the current character and its consecutive count
    max_consecutive_count = 0
    current_char = ''
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

def emb_to_seq(emb, model, tokenizer):
    decoder_input_ids = tokenizer("<pad>", return_tensors='pt').input_ids
    decoder_input_ids = decoder_input_ids.to(DEVICE)
    # Generate the output sequence using the noised hidden states from the encoder 
    output_ids = model.generate(
        input_ids=decoder_input_ids,
        encoder_outputs=(emb,),
        max_length=emb.shape[1]
    )
    output_seq = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    output_seq = output_seq.replace(" ", "")
    del decoder_input_ids, output_ids
    return output_seq

def emb_to_seq_check(emb,model,tokenizer,noise_scale=0.3,noise_step=0.1,noise_step_thresh=50,total_thresh=500,tqdm_bar:tqdm=None):
    success_flag = False
    gen_round = 0
    while((not success_flag) and noise_scale!=0):
        if(gen_round!=0):
            noised_emb = emb+torch.randn_like(emb)*noise_scale
        else:
            noised_emb = emb
        decoder_input_ids = tokenizer("<pad>", return_tensors='pt').input_ids
        decoder_input_ids = decoder_input_ids.to(DEVICE)
        # Generate the output sequence using the noised hidden states from the encoder 
        output_ids = model.generate(
            input_ids=decoder_input_ids,
            encoder_outputs=(noised_emb,),
            max_length=emb.shape[1]
        )
        output_seq = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        output_seq = output_seq.replace(" ", "")
        success_flag = (not has_consecutive_AA(output_seq)) and (not has_repeated_AA(output_seq))
        if((gen_round%noise_step_thresh)==0 and gen_round!=0):
            noise_scale+=noise_step
            # print("Increase Noise to:", noise_scale)
        if(gen_round>=total_thresh):
            break
        gen_round += 1
        if(tqdm_bar):
            post_fix = tqdm_bar.postfix.split("=")
            post_fix = {post_fix[0]:post_fix[1][:4]}
            post_fix["gen_round"] = gen_round
            post_fix["noise_scale"] = noise_scale
            tqdm_bar.set_postfix(post_fix)
    del success_flag, noised_emb, decoder_input_ids, output_ids
    return output_seq, gen_round

if __name__ == "__main__":
    embeddings = load_all_embedding()
    embedding_scale = load_embedding_scaler()
    model, tokenizer = load_model()
    for seed in range(10):
    # for seed in [4]:#0,1,9,8,7,6,5,4
        all_emb = embeddings[seed]
        pred_seqs = []
        tqdm_bar = tqdm(all_emb, total=len(all_emb))
        for pdb_id, e in tqdm_bar:
            e = torch.Tensor(np.array([embedding_scale.inverse_transform(e)])).to(DEVICE)
            tqdm_bar.set_postfix({"pdb_id":pdb_id})
            # seq = emb_to_seq(e, model, tokenizer)
            seq, gen_round = emb_to_seq_check(e, model, tokenizer,tqdm_bar=tqdm_bar)
            pred_seqs.append((pdb_id, seq, gen_round))
            del e, pdb_id, seq
        with open(SEQ_FILE.format(seed=seed), "+wb") as f:
            pickle.dump(pred_seqs, f)
        del pred_seqs
