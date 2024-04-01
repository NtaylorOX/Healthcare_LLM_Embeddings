from pathlib import Path

import typer
from allennlp.common import util as common_util
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from transformers import AutoTokenizer
import argparse

# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
SAVING = "\U0001F4BE"
HUGGING_FACE = "\U0001F917"


def main(archive_file: str, save_directory: Path) -> None:
    """Saves the model and tokenizer from an AllenNLP `archive_file` path pointing to a trained
    DeCLUTR model to a format that can be used with HuggingFace Transformers at `save_directory`."""
    save_directory = Path(save_directory)
    save_directory.parents[0].mkdir(parents=True, exist_ok=True)

    common_util.import_module_and_submodules("declutr")
    # cuda_device -1 places the model onto the CPU before saving. This avoids issues with
    # distributed models.
    overrides = "{'trainer.cuda_device': -1}"
    # print(f"archive file: {archive_file}")
    
    archive = load_archive(archive_file, overrides=overrides)
    predictor = Predictor.from_archive(archive, predictor_name="declutr")

    # print(f"predictor _model was: {predictor._model} and keys are:")
    token_embedder = predictor._model._text_field_embedder._token_embedders["tokens"]
    # print(f"token embedder was:\n {token_embedder}")
    # attrs = vars(token_embedder)
   
    # # now dump this in some way or another
    # print(', '.join("%s: %s" % item for item in attrs.items()))

    model = token_embedder.transformer_model
    # print("token embedding model name or path is: ", token_embedder.config._name_or_path)
    # # print(f"model is : {model}")
    if token_embedder.masked_language_modeling:
        # HF tokenizer is already stored
        tokenizer = token_embedder.tokenizer
    else:
        typer.secho(f"tokenizer had not been saved - reloading now given the following model_name_or_path: {token_embedder.config._name_or_path}")
        tokenizer = AutoTokenizer.from_pretrained(token_embedder.config._name_or_path)

    # Casting as a string to avoid this error: https://github.com/huggingface/transformers/pull/4650
    # Can be removed after PR is merged and Transformers is updated.
    model.save_pretrained(str(save_directory))
    tokenizer.save_pretrained(str(save_directory))

    typer.secho(
        (
            f"{SAVING} {HUGGING_FACE} Transformers compatible model saved to: {save_directory}."
            " See https://huggingface.co/transformers/model_sharing.html for instructions on"
            f" hosting the model with {HUGGING_FACE} Transformers."
        ),
        bold=True,
    )


if __name__ == "__main__":
    # typer.run(main)
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--archive_file",
                        default = "E:/saved_models/declutr/wiki/output/",
                        type=str,
                        help = "The data path to the file containing the saved model etc")
    
    parser.add_argument("--save_directory",
                        default = "E:/saved_models/declutr/wiki/output/transformer_format/",
                        type=str,
                        help = "The data path to the file containing the saved model etc")
    
    
    # create args object
    args = parser.parse_args()
    
    # now run
    main(archive_file=args.archive_file, save_directory=args.save_directory)
