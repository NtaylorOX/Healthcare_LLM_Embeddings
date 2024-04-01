#!/usr/bin/env python3
import io
import re
import os
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
import typer
from declutr.common.util import sanitize_text

import argparse
# Emoji's used in typer.secho calls
# See: https://github.com/carpedm20/emoji/blob/master/emoji/unicode_codes.py"
SAVING = "\U0001F4BE"
DOWNLOAD = "\U00002B07"



def _write_output_to_disk(text: List[str], output_filepath: Path) -> None:
    """Writes a list of documents, `text`, to the file `output_filepath`, one document per line."""
    # Create the directory path if it doesn't exist
    output_filepath = Path(output_filepath)
    output_filepath.parents[0].mkdir(parents=True, exist_ok=True)

    with open(output_filepath, "w", encoding = 'utf-8') as f:
        # TODO (John): In the future, it might make sense to both batch and shard:
        # 1) Batch, meaning write batches of documents to a file as opposed to 1 at a time
        # 2) Shard, meaning break a file up into shard_size // len(text) files, and return a
        #    directory instead. Loading a dataset like this is supported in AllenNLP (see:
        #    https://docs.allennlp.org/master/api/data/dataset_readers/sharded_dataset_reader/)
        with typer.progressbar(text, label="Writing to disk") as progress:
            for doc in progress:                
                f.write(doc.strip() + "\n")
    typer.secho(
        f"{SAVING} {len(text)} preprocessed documents saved to: {output_filepath}",
        bold=True,
    )


def main(
    input_filepath: Path,
    output_filepath: Path,
    segment_sentences: bool = False,
    lowercase: bool = False,
    min_length: Optional[int] = None,
    max_instances: Optional[int] = None,
    pretrained_model_name_or_path: Optional[str] = None,
) -> None:
    """Lightly preprocess the text dataset. If `min_length is not None`, only documents
    with at least this many tokens are retained. If `pretrained_model_name_or_path` is not None, the
    tokenizer will be loaded as `AutoTokenizer.from_pretrained(pretrained_model_name_or_path)`
    using the HuggingFace Transformers library. Otherwise `str.split()` is used. This argument has
    no effect if `min-length is None`. If `segment_sentences` is provided, individual sentences
    will be returned instead of documents. You must have the `"en_core_web_sm"` spacy model
    installed to segment sentences.
    """
    # Setup the pre-trained tokenizer, if specified
    if min_length is not None:
        if pretrained_model_name_or_path is not None:
            # Import transformers here to prevent ImportError errors if the
            # user doesn't want to use it.
            from transformers import AutoTokenizer
            typer.secho(f"Will be tokenizing with: {pretrained_model_name_or_path}")
            tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path).tokenize
        else:
            tokenizer = lambda x: x.split()  # noqa
    else:
        tokenizer = None

    # Setup spacy lang object if we are segmenting sentences
    if segment_sentences:
        import spacy

        nlp = spacy.load("en_core_web_sm", disable=["ner"])

    # # Download WikiText-103
    # r = requests.get(WIKITEXT_103_URL, stream=True)
    # z = zipfile.ZipFile(io.BytesIO(r.content))
    # partition_filenames = z.namelist()[1:]
    # typer.secho(f"{DOWNLOAD} Downloaded WikiText-103", bold=True)    
 

    preprocessed_documents: List[str] = []
    
    # load in the one big training file
    with open(input_filepath, 'r') as f: 
        text = f.readlines()

       
        # Strip out subtitles and split the text into documents
        documents = text
        
        if segment_sentences:
            documents = (sent.text for doc in documents for sent in nlp(doc).sents)  # type: ignore

        with typer.progressbar(
            documents, length=max_instances, label=typer.style("Preprocessing text", bold=True)
        ) as progress:
            for doc in progress:
                doc = sanitize_text(doc, lowercase=lowercase)
                # print(f"Doc processed is: {doc}")
                if not doc:
                    continue

                # Retain documents if the length of their shortest document is
                # equal to or greater than the minimum specified length
                if tokenizer is not None:
                    # We add a space in front of the text in order to achieve consistant tokenization with
                    # certain tokenizers, e.g. the BPE tokenizer used by RoBERTa, GPT and others.
                    # See: https://github.com/huggingface/transformers/issues/1196
                    if pretrained_model_name_or_path is not None:
                        doc = f" {doc.lstrip()}"
                    num_tokens = len(tokenizer(doc))
                    # print(f"num tokens:{num_tokens}")
                    
                
                    if min_length and num_tokens < min_length:                        
                        continue

                if max_instances and len(preprocessed_documents) >= max_instances:
                    break
                preprocessed_documents.append(doc)
                progress.update(1)

    _write_output_to_disk(preprocessed_documents, output_filepath)

    # ensure file close
    f.close()

if __name__ == "__main__":
    



    # typer.run(main)
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--input_filepath",
                        default = "/mnt/sdg/niallt/mimic_iii/processed/HADM_ID_split/train_250000.txt",
                        type=str,
                        help = "The data path to the file containing the saved model etc")
    
    parser.add_argument("--save_directory",
                        default = "/mnt/sdg/niallt/mimic_iii/processed/HADM_ID_split/declutr/",
                        type=str,
                        help = "The data path to the file containing the saved model etc")
    parser.add_argument("--segment_sentences",
                        default = False,
                        type=bool,
                        help = "Whether to segment the documents into sentences using spaCy")
    parser.add_argument("--min_length",
                        default = 128,
                        type=int,
                        help = "The minimum number of tokens a document should have to be kept. If none will be calculated based on num_anchors * max_span_len * 2")
  
    parser.add_argument("--max_instances",
                        default = None,
                        type=int,
                        help = "The total maximum of documents to keep")
    parser.add_argument("--pretrained_model_name_or_path", # roberta-base | johngiorgi/declutr-sci-base
                        default = "roberta-base", # if None will do str.split()
                        type=str,
                        help = "The name of HF model")
        
    
    # create args object
    args = parser.parse_args()    
            
    # update save_dir based on the min_length
    # remember it is recommended by the original authors that min_length = num_anchors * max_span_len * 2
    # the wiki dataset had much longer documents - so we may just use one anchor for now with max_span_len of 128
    

    # args.min_length # should be one of 16/32/64/128/256/512/1024/2048
    

    # change save directory based on whether roberta-base or sci-bert was used to tokenizer. 
    #NOTE THIS IS SUPER CRUDE AND SHOULD BE UPDATED TO BE MUCH MORE ROBUST/DYNAYMIC BASED ON MODEL NAME
    
    if "sci-base" in args.pretrained_model_name_or_path:
        typer.secho(f"Got bert-sci-base model with a special tokenizer!")
        args.save_directory = f"{args.save_directory}/bert-sci-base/"
    
    
    typer.secho(f"minimum length is: {args.min_length}", bold=True)
    
    if args.max_instances:
        output_filepath = f"{args.save_directory}/sample_{args.max_instances}/min_{args.min_length}/train.txt"
        
    else:           
        
        # update the output directory based on the min_length and num_anchors
        output_filepath = f"{args.save_directory}/sample_250000/min_{args.min_length}/train.txt"
        
        

    
    # now run with arguments required by main
    main(input_filepath=args.input_filepath,
            output_filepath = output_filepath, # NOTE this is not based on the provided args
            segment_sentences= args.segment_sentences,
            min_length = args.min_length, 
            max_instances = args.max_instances,                        
            pretrained_model_name_or_path = args.pretrained_model_name_or_path)