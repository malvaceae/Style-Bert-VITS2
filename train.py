import argparse
import json
import subprocess
import sys
from distutils.util import strtobool
from pathlib import Path

import yaml

from webui_train import (
    get_path,
    preprocess_all,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()

    with Path(args.input_dir, "config/hyperparameters.json").open() as f:
        hyperparameters = json.load(f)

    use_jp_extra, batch_size, epochs, save_every_steps, normalize, trim = (
        bool(strtobool(hyperparameters.get("use_jp_extra", "True"))),
        int(hyperparameters.get("batch_size", "4")),
        int(hyperparameters.get("epochs", "100")),
        int(hyperparameters.get("save_every_steps", "1000")),
        bool(strtobool(hyperparameters.get("normalize", "False"))),
        bool(strtobool(hyperparameters.get("trim", "False"))),
    )

    for model in Path(args.input_dir, "data/train").iterdir():
        (
            dataset_path,
            lbl_path,
            _,
            _,
            config_path,
        ) = get_path(model.name)

        subprocess.run(
            [
                sys.executable,
                "slice.py",
                "--input_dir",
                model,
                "--output_dir",
                Path(dataset_path, "raw"),
            ],
            check=True,
        )

        subprocess.run(
            [
                sys.executable,
                "transcribe.py",
                "--input_dir",
                Path(dataset_path, "raw"),
                "--output_file",
                lbl_path,
                "--speaker_name",
                model.name,
                "--compute_type",
                "float16",
            ],
            check=True,
        )

        success, message = preprocess_all(
            model_name=model.name,
            batch_size=batch_size,
            epochs=epochs,
            save_every_steps=save_every_steps,
            num_processes=2,
            normalize=normalize,
            trim=trim,
            freeze_EN_bert=False,
            freeze_JP_bert=False,
            freeze_ZH_bert=False,
            freeze_style=False,
            use_jp_extra=use_jp_extra,
            val_per_lang=0,
            log_interval=200,
        )

        if not success:
            raise RuntimeError(message)

        with Path("default_config.yml").open() as f:
            config = yaml.safe_load(f)

        config["model_name"] = model.name

        with Path("config.yml").open(mode="w") as f:
            yaml.dump(config, f, allow_unicode=True)

        subprocess.run(
            [
                sys.executable,
                "train_ms_jp_extra.py" if use_jp_extra else "train_ms.py",
                "--config",
                config_path,
                "--model",
                dataset_path,
                "--assets_root",
                args.model_dir,
            ],
            check=True,
        )
