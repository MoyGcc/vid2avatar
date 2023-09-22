from v2a_model import V2AModel
from lib.datasets import create_dataset
import hydra
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
import os
import glob
# from pytorch_lightning.profilers import PyTorchProfiler, SimpleProfiler, AdvancedProfiler
from torch.profiler import profile, record_function, ProfilerActivity


@hydra.main(config_path="confs", config_name="base")
def main(opt):
    pl.seed_everything(42)
    print("Working dir:", os.getcwd())

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="checkpoints/",
        filename="{epoch:04d}-{loss}",
        save_on_train_epoch_end=True,
        save_last=True,
    )
    logger = WandbLogger(project=opt.project_name, name=f"{opt.exp}/{opt.run}")

    # profiler = AdvancedProfiler(dirpath="/media/AI", filename="advanced.txt")
    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        callbacks=[checkpoint_callback],
        logger=logger,
        log_every_n_steps=1,
        # profiler=profiler,
    )

    model = V2AModel(opt)
    checkpoint = sorted(glob.glob("checkpoints/*.ckpt"))[-1]
    testset = create_dataset(opt.dataset.metainfo, opt.dataset.test)

    # profiler
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    #     with record_function("model_inference"):
    trainer.test(model, testset, ckpt_path=checkpoint)
    
    # print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))
    # prof.export_chrome_trace("/media/AI/trace.json")



if __name__ == "__main__":
    main()
