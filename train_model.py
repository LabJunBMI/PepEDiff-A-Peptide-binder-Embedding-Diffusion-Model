import torch
import pickle
import pytorch_lightning as pl
from transformers import BertConfig
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.utilities.seed import seed_everything

from dataset import LigandBindingSiteDataset
from model import ConditionalBertForDiffusion

MODEL_PATH = "./model.pt"
TRAIN_DATA_FILE = "./data/seq_emb_1024_no_x_z_train.pkl"
VAL_DATA_FILE = "./data/seq_emb_1024_no_x_z_val.pkl"

GPU_ID = [0]
NUM_THREAD = 16

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
    "random_seed":0,
    "feature_size": 1024
}

def get_dataloader():
    # with open(SPLIT_FILE, "rb") as f:
    #     data_split = pickle.load(f)
    print("Loading Training Set")
    train_ds = LigandBindingSiteDataset(
        TRAIN_DATA_FILE, #"train", data_split
    )
    train_dataloader = DataLoader(
        dataset=train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,  # Shuffle only train loader
        num_workers=NUM_THREAD,
        prefetch_factor=2
    )
    print("Loading Validation Set")
    val_ds = LigandBindingSiteDataset(
        VAL_DATA_FILE, #"val", data_split
    )
    val_dataloader = DataLoader(
        dataset=val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,  # Shuffle only train loader
        num_workers=NUM_THREAD,
        prefetch_factor=2
    )
    return train_dataloader, val_dataloader

def train_model(
    bert_config: BertConfig,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
):
    checkpoint_callback = ModelCheckpoint(
        dirpath="./",
        monitor="val_loss",  # Monitor the validation loss
        filename="best_val_model",  # Filename template
        save_top_k=1,  # Only save the best model
        mode="max",  # Save the model with the highest validation loss
    )

    model = ConditionalBertForDiffusion(
        feature_size=CONFIG["feature_size"],
        bert_config=bert_config,
        epochs=CONFIG["max_epochs"],
        lr_scheduler=CONFIG["lr_scheduler"],
        l2_lambda=CONFIG["l2_norm"],
        steps_per_epoch=len(train_dataloader),
        learning_rate=CONFIG["lr"],
    )
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model has {num_params} trainable parameters")
    trainer = pl.Trainer(
        default_root_dir="./",
        gradient_clip_val=CONFIG["gradient_clip"],
        callbacks=[checkpoint_callback],
        min_epochs=CONFIG["min_epochs"],
        max_epochs=CONFIG["max_epochs"],
        check_val_every_n_epoch=1,
        log_every_n_steps=30,
        accelerator="gpu",
        devices=GPU_ID,
        # move_metrics_to_cpu=False,  # Saves memory
    )
    print("Start training")
    trainer.fit(
        model=model,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    return trainer, model

if __name__ == "__main__":
    seed_everything(CONFIG["random_seed"])
    torch.set_float32_matmul_precision("medium")
    torch.set_num_threads(NUM_THREAD)
    train_dataloader, val_dataloader = get_dataloader()
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

    trainer, model = train_model(
        bert_config, train_dataloader, val_dataloader
    )
    torch.save(model.state_dict(), MODEL_PATH)