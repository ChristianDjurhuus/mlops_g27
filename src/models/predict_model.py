from omegaconf import DictConfig
import torch
from datasets import load_metric
from src.models.model import ImdbTransformer
from src.data.load_data import ImdbDataModule
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

@hydra.main(config_path="config", config_name="default_config.yaml")
def evaluate(cfg:DictConfig):
    print("Evaluating")
    metric = load_metric("accuracy")
    # initializing wandb logging
    # Load Data
    cfg = cfg.experiment
    dm = ImdbDataModule(
        model_name=cfg.model,
        data_path=to_absolute_path(cfg.data_path),
        batch_size=cfg.hyper_param["batch_size"],
        debug=cfg.debug_mode,
        seed=cfg.seed,
    )
    dm.prepare_data()
    dm.setup("predict")
    test_dataloader=dm.test_dataloader()

    model = ImdbTransformer(
        model_name="bert-base-cased", learning_rate=0.001, batch_size=1,
    )
    model.load_from_checkpoint(
        to_absolute_path("models/models/3jnhnpvp/checkpoints/epoch=0-step=0.ckpt")
    )

    # if there is a GPU, the pytorch lightning will move data to cuda automaticly (i guess.)
    # device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # print(device)
    # model.to(device)
    model.eval()
    with torch.no_grad():
        for batch in test_dataloader:
            # batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)
            metric.add_batch(predictions=predictions, references=batch["labels"])
            print(f'Accuracy: {metric.compute().get("accuracy")*100}%')


if __name__ == "__main__":
    evaluate()
