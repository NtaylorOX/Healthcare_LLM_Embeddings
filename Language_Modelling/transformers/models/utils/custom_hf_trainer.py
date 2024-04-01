# works for transformers 4.27.3
from transformers import (
    PreTrainedModel,
    RobertaPreTrainedModel, 
    RobertaModel, 
    PretrainedConfig, 
    RobertaConfig, 
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,AutoModelForCausalLM,AutoModelForMaskedLM, 
    AutoConfig, 
    AutoModel,
    AutoTokenizer,
    RobertaPreTrainedModel,
    BertForPreTraining,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)

from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    BaseModelOutputWithPoolingAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    NextSentencePredictorOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)

from transformers.trainer_utils import (
    PREFIX_CHECKPOINT_DIR,
    BestRun,
    EvalLoopOutput,
    EvalPrediction,
    FSDPOption,
    HPSearchBackend,
    HubStrategy,
    IntervalStrategy,
    PredictionOutput,
    RemoveColumnsCollator,
    ShardedDDPOption,
    TrainerMemoryTracker,
    TrainOutput,
    default_compute_objective,
    default_hp_space,
    denumpify_detensorize,
    enable_full_determinism,
    find_executable_batch_size,
    get_last_checkpoint,
    has_length,
    number_of_arguments,
    seed_worker,
    set_seed,
    speed_metrics,
)

from transformers.utils import logging

from transformers.configuration_utils import PretrainedConfig
from transformers.data.data_collator import DataCollator, DataCollatorWithPadding, default_data_collator
from transformers.debug_utils import DebugOption, DebugUnderflowOverflow
from transformers.deepspeed import deepspeed_init, is_deepspeed_zero3_enabled
from transformers.dependency_versions_check import dep_version_check
from transformers.modelcard import TrainingSummary
from transformers.modeling_utils import PreTrainedModel, load_sharded_checkpoint, unwrap_model
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES, MODEL_MAPPING_NAMES
from transformers.optimization import Adafactor, get_scheduler
# from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS, is_torch_greater_or_equal_than_1_10, is_torch_less_than_1_11
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.trainer_callback import (
    CallbackHandler,
    DefaultFlowCallback,
    PrinterCallback,
    ProgressCallback,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_pt_utils import (
    DistributedLengthGroupedSampler,
    DistributedSamplerWithLoop,
    DistributedTensorGatherer,
    IterableDatasetShard,
    LabelSmoother,
    LengthGroupedSampler,
    SequentialDistributedSampler,
    ShardSampler,
    distributed_broadcast_scalars,
    distributed_concat,
    find_batch_size,
    get_module_class_from_name,
    get_parameter_names,
    nested_concat,
    nested_detach,
    nested_numpify,
    nested_truncate,
    nested_xla_mesh_reduce,
    reissue_pt_warnings,
)
import time

from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

from transformers.utils.generic import (
    ContextManagers,
    ExplicitEnum,
    ModelOutput,
    PaddingStrategy,
    TensorType,
    cached_property,
    can_return_loss,
    expand_dims,
    find_labels,
    flatten_dict,
    is_jax_tensor,
    is_numpy_array,
    is_tensor,
    is_tf_tensor,
    is_torch_device,
    is_torch_dtype,
    is_torch_tensor,
    reshape,
    squeeze,
    tensor_size,
    to_numpy,
    to_py_obj,
    transpose,
    working_or_temp_dir,
)
from transformers.utils.hub import (
    CLOUDFRONT_DISTRIB_PREFIX,
    DISABLE_TELEMETRY,
    HF_MODULES_CACHE,
    HUGGINGFACE_CO_PREFIX,
    HUGGINGFACE_CO_RESOLVE_ENDPOINT,
    PYTORCH_PRETRAINED_BERT_CACHE,
    PYTORCH_TRANSFORMERS_CACHE,
    S3_BUCKET_PREFIX,
    TRANSFORMERS_CACHE,
    TRANSFORMERS_DYNAMIC_MODULE_NAME,
    EntryNotFoundError,
    PushToHubMixin,
    RepositoryNotFoundError,
    RevisionNotFoundError,
    cached_file,
    default_cache_path,
    define_sagemaker_information,
    download_url,
    extract_commit_hash,
    get_cached_models,
    get_file_from_repo,
    get_full_repo_name,
    has_file,
    http_user_agent,
    is_offline_mode,
    is_remote_url,
    move_cache,
    send_example_telemetry,
)
from transformers.utils.import_utils import (
    ENV_VARS_TRUE_AND_AUTO_VALUES,
    ENV_VARS_TRUE_VALUES,
    TORCH_FX_REQUIRED_VERSION,
    USE_JAX,
    USE_TF,
    USE_TORCH,
    DummyObject,
    OptionalDependencyNotAvailable,
    _LazyModule,
    ccl_version,
    is_accelerate_available,
    is_apex_available,
    is_bitsandbytes_available,
    is_bs4_available,
    is_coloredlogs_available,
    is_cython_available,
    is_datasets_available,
    is_decord_available,
    is_detectron2_available,
    is_faiss_available,
    is_flax_available,
    is_ftfy_available,
    is_in_notebook,
    is_ipex_available,
    is_jumanpp_available,
    is_kenlm_available,
    is_keras_nlp_available,
    is_librosa_available,   
    is_natten_available,
    is_ninja_available,
    is_onnx_available,
    is_pandas_available,
    is_phonemizer_available,
    is_protobuf_available,
    is_psutil_available,
    is_py3nvml_available,
    is_pyctcdecode_available,
    is_pytesseract_available,
    is_pytorch_quantization_available,
    is_rjieba_available,
    is_sacremoses_available,
    is_safetensors_available,
    is_sagemaker_dp_enabled,
    is_sagemaker_mp_enabled,
    is_scipy_available,
    is_sentencepiece_available,
    is_sklearn_available,
    is_soundfile_availble,
    is_spacy_available,
    is_speech_available,
    is_sudachi_available,
    is_tensorflow_probability_available,
    is_tensorflow_text_available,
    is_tf2onnx_available,
    is_tf_available,
    is_timm_available,
    is_tokenizers_available,
    is_torch_available,
    is_torch_bf16_available,
    is_torch_bf16_cpu_available,
    is_torch_bf16_gpu_available,
    is_torch_compile_available,
    is_torch_cuda_available,
    is_torch_fx_available,
    is_torch_fx_proxy,
    is_torch_neuroncore_available,
    # is_torch_onnx_dict_inputs_support_available, # worked in 4.27
    is_torch_tensorrt_fx_available,
    is_torch_tf32_available,
    is_torch_tpu_available,
    is_torchaudio_available,
    is_torchdistx_available,
    is_torchdynamo_available,
    is_training_run_on_sagemaker,
    is_vision_available,
    requires_backends,
    torch_only_method,
    torch_version,
)

# from transformers.utils.generic import can_return_loss
# custom

logger = logging.get_logger(__name__)
logging.set_verbosity_info()
import math
from typing import List, Optional, Tuple, Union, NamedTuple, Any

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from datasets import load_dataset, load_metric # list_datasets, load_from_disk, DatasetDict, Dataset, load_dataset_builder
import evaluate # this weirdly loads something onto the GPU and will cause OOM on python3.9
import pandas as pd
from tqdm import tqdm
import numpy as np
import copy
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
# from sentence_transformers.losses import CosineSimilarityLoss


import os

class CustomHFTrainer(Trainer):
    

    
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
            """
            Perform a training step on a batch of inputs.
            Subclass and override to inject custom behavior.
            Args:
                model (`nn.Module`):
                    The model to train.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.
                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
            Return:
                `torch.Tensor`: The tensor with training loss on this batch.
            """
            

            
            model.train()
            inputs = self._prepare_inputs(inputs)
 
            
            with self.compute_loss_context_manager():
                loss, mlm_loss, cls_loss = self.compute_loss(model, inputs)
                

            if self.args.n_gpu > 1:
                #TODO - double check this is working correctly i.e. that it isn't the combiend losses
                # also need to test whether MLM only is actually working - as the performance of both so far looks really poor
                # add contrastive loss handling
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
                if mlm_loss is not None:
                    mlm_loss = mlm_loss.mean()
                if cls_loss is not None:
                    cls_loss = cls_loss.mean()
            if self.args.gradient_accumulation_steps > 1 and self.deepspeed is None:
                # deepspeed handles loss scaling by gradient_accumulation_steps in its `backward`
                loss = loss / self.args.gradient_accumulation_steps
                if mlm_loss is not None:
                    mlm_loss = mlm_loss / self.args.gradient_accumulation_steps
                if cls_loss is not None:
                    cls_loss = cls_loss / self.args.gradient_accumulation_steps
            if self.do_grad_scaling:
                self.scaler.scale(loss).backward()
            elif self.use_apex:
                with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                    scaled_loss.backward()
            elif self.deepspeed:
                # loss gets scaled under gradient_accumulation_steps in deepspeed
                loss = self.deepspeed.backward(loss)
            else:
                loss.backward()
            # print(f"mlm_loss:{mlm_loss} \n seq_cls loss: {cls_loss}")
            return loss.detach(), mlm_loss, cls_loss

    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        # print(f"inputs are: {inputs}")
        if self.label_smoother is not None and "labels" in inputs:           
            labels = inputs.pop("labels")
            if "category_label" in inputs:
                cat_labels = inputs.pop("category_label")
        else:
            labels = None
        # set category labels if we have any
        # if "category_label" in inputs:
        #     cat_labels = inputs["category_label"]
            # print(f"got cat labels: {cat_labels}")
        outputs = model(**inputs)
        # print(f"outputs are: {outputs}")
        
        # print(f"shape of embeddings is:", outputs["seq_embedding"].shape)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]        
        if labels is not None:            
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
            mlm_loss = outputs["mlm_loss"] if isinstance(outputs, dict) else outputs[1]
            cls_loss = outputs["cls_loss"] if isinstance(outputs, dict) else outputs[2]
            
            # print(f"loss we got back was: {loss}")
            #TODO - add contrastive loss calc here.
            # if cat_labels is not None and "seq_embedding" in outputs:
            #     # print(f"got category labels and sequence embeddings - computing supervised contrastive loss")                
            #     seq_embeds = outputs["seq_embedding"]                
            # also get the separate losses for each
        if return_outputs:
            # print(f"Return output is: {outputs}")
            return (loss, mlm_loss, cls_loss, outputs)
        else:
            return loss, mlm_loss, cls_loss
    
    
    def _inner_training_loop(
            self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
        ):        
        self._train_batch_size = batch_size
        # Data loader and number of training steps
        train_dataloader = self.get_train_dataloader()

        # Setting up training control variables:
        # number of training epochs: num_train_epochs
        # number of training steps per epoch: num_update_steps_per_epoch
        # total number of training steps to execute: max_steps
        total_train_batch_size = args.train_batch_size * args.gradient_accumulation_steps * args.world_size        
        len_dataloader = None
        if has_length(train_dataloader):
            len_dataloader = len(train_dataloader)
            num_update_steps_per_epoch = len_dataloader // args.gradient_accumulation_steps
            num_update_steps_per_epoch = max(num_update_steps_per_epoch, 1)
            num_examples = self.num_examples(train_dataloader)
            if args.max_steps > 0:
                max_steps = args.max_steps
                num_train_epochs = args.max_steps // num_update_steps_per_epoch + int(
                    args.max_steps % num_update_steps_per_epoch > 0
                )
                # May be slightly incorrect if the last batch in the training dataloader has a smaller size but it's
                # the best we can do.
                num_train_samples = args.max_steps * total_train_batch_size
            else:
                max_steps = math.ceil(args.num_train_epochs * num_update_steps_per_epoch)
                num_train_epochs = math.ceil(args.num_train_epochs)
                num_train_samples = self.num_examples(train_dataloader) * args.num_train_epochs
        elif args.max_steps > 0:  # Rely on max_steps when dataloader does not have a working size
            max_steps = args.max_steps
            # Setting a very large number of epochs so we go as many times as necessary over the iterator.
            num_train_epochs = sys.maxsize
            num_update_steps_per_epoch = max_steps
            num_examples = total_train_batch_size * args.max_steps
            num_train_samples = args.max_steps * total_train_batch_size
        else:
            raise ValueError(
                "args.max_steps must be set to a positive value if dataloader does not have a length, was"
                f" {args.max_steps}"
            )

        if DebugOption.UNDERFLOW_OVERFLOW in self.args.debug:
            if self.args.n_gpu > 1:
                # nn.DataParallel(model) replicates the model, creating new variables and module
                # references registered here no longer work on other gpus, breaking the module
                raise ValueError(
                    "Currently --debug underflow_overflow is not supported under DP. Please use DDP"
                    " (torch.distributed.launch)."
                )
            else:
                debug_overflow = DebugUnderflowOverflow(self.model)  # noqa

        delay_optimizer_creation = (
            self.sharded_ddp is not None
            and self.sharded_ddp != ShardedDDPOption.SIMPLE
            or is_sagemaker_mp_enabled()
            or self.fsdp is not None
        )
        if args.deepspeed:                
            deepspeed_engine, optimizer, lr_scheduler = deepspeed_init(
                self, num_training_steps=max_steps, resume_from_checkpoint=resume_from_checkpoint
            )
            self.model = deepspeed_engine.module
            self.model_wrapped = deepspeed_engine
            self.deepspeed = deepspeed_engine
            self.optimizer = optimizer
            self.lr_scheduler = lr_scheduler
        elif not delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        self.state = TrainerState()
        self.state.is_hyper_param_search = trial is not None

        # Activate gradient checkpointing if needed
        if args.gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        model = self._wrap_model(self.model_wrapped)

        if is_sagemaker_mp_enabled() and resume_from_checkpoint is not None:
            self._load_from_checkpoint(resume_from_checkpoint, model)

        # for the rest of this function `model` is the outside model, whether it was wrapped or not
        if model is not self.model:
            self.model_wrapped = model

        if delay_optimizer_creation:
            self.create_optimizer_and_scheduler(num_training_steps=max_steps)

        # Check if saved optimizer or scheduler states exist
        self._load_optimizer_and_scheduler(resume_from_checkpoint)

        # important: at this point:
        # self.model         is the Transformers Model
        # self.model_wrapped is DDP(Transformers Model), Deepspeed(Transformers Model), etc.
        
        # Train!
        logger.info("***** Running training rabblleee *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Num Epochs = {num_train_epochs}")
        logger.info(f"  Instantaneous batch size per device = {args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")
        logger.info(
            f"  Number of trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad)}"
        )

        self.state.epoch = 0
        start_time = time.time()
        epochs_trained = 0
        steps_trained_in_current_epoch = 0
        steps_trained_progress_bar = None

        # Check if continuing training from a checkpoint
        if resume_from_checkpoint is not None and os.path.isfile(
            os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME)
        ):
            self.state = TrainerState.load_from_json(os.path.join(resume_from_checkpoint, TRAINER_STATE_NAME))
            epochs_trained = self.state.global_step // num_update_steps_per_epoch
            if not args.ignore_data_skip:
                steps_trained_in_current_epoch = self.state.global_step % (num_update_steps_per_epoch)
                steps_trained_in_current_epoch *= args.gradient_accumulation_steps
            else:
                steps_trained_in_current_epoch = 0

            logger.info("  Continuing training from checkpoint, will skip to saved global_step")
            logger.info(f"  Continuing training from epoch {epochs_trained}")
            logger.info(f"  Continuing training from global step {self.state.global_step}")
            if not args.ignore_data_skip:
                logger.info(
                    f"  Will skip the first {epochs_trained} epochs then the first {steps_trained_in_current_epoch} "
                    "batches in the first epoch. If this takes a lot of time, you can add the `--ignore_data_skip` "
                    "flag to your launch command, but you will resume the training on data already seen by your model."
                )
                if self.is_local_process_zero() and not args.disable_tqdm:
                    steps_trained_progress_bar = tqdm(total=steps_trained_in_current_epoch)
                    steps_trained_progress_bar.set_description("Skipping the first batches")

        # Update the references
        self.callback_handler.model = self.model
        self.callback_handler.optimizer = self.optimizer
        self.callback_handler.lr_scheduler = self.lr_scheduler
        self.callback_handler.train_dataloader = train_dataloader
        if self.hp_name is not None and self._trial is not None:
            # use self._trial because the SigOpt/Optuna hpo only call `_hp_search_setup(trial)` instead of passing trial
            # parameter to Train when using DDP.
            self.state.trial_name = self.hp_name(self._trial)
        if trial is not None:
            assignments = trial.assignments if self.hp_search_backend == HPSearchBackend.SIGOPT else trial
            self.state.trial_params = hp_params(assignments)
        else:
            self.state.trial_params = None
        # This should be the same if the state has been saved but in case the training arguments changed, it's safer
        # to set this after the load.
        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        # tr_loss is a tensor to avoid synchronization of TPUs through .item()
        tr_loss = torch.tensor(0.0).to(args.device)
        mlm_loss = torch.tensor(0.0).to(args.device)
        cls_loss = torch.tensor(0.0).to(args.device)
        # _total_loss_scalar is updated everytime .item() has to be called on tr_loss and stores the sum of all losses
        self._total_loss_scalar = 0.0
        self._total_mlm_loss_scalar = 0.0
        self._total_cls_loss_scalar = 0.0
        self._globalstep_last_logged = self.state.global_step
        model.zero_grad()

        self.control = self.callback_handler.on_train_begin(args, self.state, self.control)

        # Skip the first epochs_trained epochs to get the random state of the dataloader at the right point.
        if not args.ignore_data_skip:
            for epoch in range(epochs_trained):
                is_random_sampler = hasattr(train_dataloader, "sampler") and isinstance(
                    train_dataloader.sampler, RandomSampler
                )
                if is_torch_less_than_1_11 or not is_random_sampler:
                    # We just need to begin an iteration to create the randomization of the sampler.
                    # That was before PyTorch 1.11 however...
                    for _ in train_dataloader:
                        break
                else:
                    # Otherwise we need to call the whooooole sampler cause there is some random operation added
                    # AT THE VERY END!
                    _ = list(train_dataloader.sampler)

        for epoch in range(epochs_trained, num_train_epochs):
            if isinstance(train_dataloader, DataLoader) and isinstance(train_dataloader.sampler, DistributedSampler):
                train_dataloader.sampler.set_epoch(epoch)
            elif hasattr(train_dataloader, "dataset") and isinstance(train_dataloader.dataset, IterableDatasetShard):
                train_dataloader.dataset.set_epoch(epoch)

            if is_torch_tpu_available():
                parallel_loader = pl.ParallelLoader(train_dataloader, [args.device]).per_device_loader(args.device)
                epoch_iterator = parallel_loader
            else:
                epoch_iterator = train_dataloader

            # Reset the past mems state at the beginning of each epoch if necessary.
            if args.past_index >= 0:
                self._past = None

            steps_in_epoch = (
                len(epoch_iterator)
                if len_dataloader is not None
                else args.max_steps * args.gradient_accumulation_steps
            )
            self.control = self.callback_handler.on_epoch_begin(args, self.state, self.control)

            if epoch == epochs_trained and resume_from_checkpoint is not None and steps_trained_in_current_epoch == 0:
                self._load_rng_state(resume_from_checkpoint)

            step = -1       
            for step, inputs in enumerate(epoch_iterator):

                # Skip past any already trained steps if resuming training
                if steps_trained_in_current_epoch > 0:
                    steps_trained_in_current_epoch -= 1
                    if steps_trained_progress_bar is not None:
                        steps_trained_progress_bar.update(1)
                    if steps_trained_in_current_epoch == 0:
                        self._load_rng_state(resume_from_checkpoint)
                    continue
                elif steps_trained_progress_bar is not None:
                    steps_trained_progress_bar.close()
                    steps_trained_progress_bar = None

                if step % args.gradient_accumulation_steps == 0:
                    self.control = self.callback_handler.on_step_begin(args, self.state, self.control)

                if (
                    ((step + 1) % args.gradient_accumulation_steps != 0)
                    and args.local_rank != -1
                    and args._no_sync_in_gradient_accumulation
                ):
                    # Avoid unnecessary DDP synchronization since there will be no backward pass on this example.
                    with model.no_sync():
                        #TODO - add mlm_loss and seq_loss handling
                        tr_loss_step, mlm_loss_step, cls_loss_step = self.training_step(model, inputs)
                else:
                    #TODO - add mlm_loss and seq_loss handling
                    tr_loss_step, mlm_loss_step, cls_loss_step = self.training_step(model, inputs)
                ### handle the total train loss 
                if (
                    args.logging_nan_inf_filter
                    and not is_torch_tpu_available()
                    and (torch.isnan(tr_loss_step) or torch.isinf(tr_loss_step))
                ):
                    # if loss is nan or inf simply add the average of previous logged losses
                    tr_loss += tr_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                else:
                    tr_loss += tr_loss_step
                ### handle the mlm_loss
                if mlm_loss_step is not None: 
                   
                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(mlm_loss_step) or torch.isinf(mlm_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        mlm_loss += mlm_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        mlm_loss += mlm_loss_step
                    
                ### handle the cls_loss
                ### handle the total train loss 
                if cls_loss_step is not None:                    
                    if (
                        args.logging_nan_inf_filter
                        and not is_torch_tpu_available()
                        and (torch.isnan(cls_loss_step) or torch.isinf(cls_loss_step))
                    ):
                        # if loss is nan or inf simply add the average of previous logged losses
                        cls_loss += cls_loss / (1 + self.state.global_step - self._globalstep_last_logged)
                    else:
                        cls_loss += cls_loss_step                   
                
                
                
                self.current_flos += float(self.floating_point_ops(inputs))

                # Optimizer step for deepspeed must be called on every step regardless of the value of gradient_accumulation_steps
                if self.deepspeed:
                    self.deepspeed.step()

                if (step + 1) % args.gradient_accumulation_steps == 0 or (
                    # last step in epoch but step is always smaller than gradient_accumulation_steps
                    steps_in_epoch <= args.gradient_accumulation_steps
                    and (step + 1) == steps_in_epoch
                ):
                    # Gradient clipping
                    if args.max_grad_norm is not None and args.max_grad_norm > 0 and self.deepspeed is None:
                        # deepspeed does its own clipping

                        if self.do_grad_scaling:
                            # Reduce gradients first for XLA
                            if is_torch_tpu_available():
                                gradients = xm._fetch_gradients(self.optimizer)
                                xm.all_reduce("sum", gradients, scale=1.0 / xm.xrt_world_size())
                            # AMP: gradients need unscaling
                            self.scaler.unscale_(self.optimizer)

                        if is_sagemaker_mp_enabled() and args.fp16:
                            self.optimizer.clip_master_grads(args.max_grad_norm)
                        elif hasattr(self.optimizer, "clip_grad_norm"):
                            # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                            self.optimizer.clip_grad_norm(args.max_grad_norm)
                        elif hasattr(model, "clip_grad_norm_"):
                            # Some models (like FullyShardedDDP) have a specific way to do gradient clipping
                            model.clip_grad_norm_(args.max_grad_norm)
                        else:
                            # Revert to normal clipping otherwise, handling Apex or full precision
                            nn.utils.clip_grad_norm_(
                                amp.master_params(self.optimizer) if self.use_apex else model.parameters(),
                                args.max_grad_norm,
                            )

                    # Optimizer step
                    optimizer_was_run = True
                    if self.deepspeed:
                        pass  # called outside the loop
                    elif is_torch_tpu_available():
                        if self.do_grad_scaling:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                        else:
                            xm.optimizer_step(self.optimizer)
                    elif self.do_grad_scaling:
                        scale_before = self.scaler.get_scale()
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        scale_after = self.scaler.get_scale()
                        optimizer_was_run = scale_before <= scale_after
                    else:
                        self.optimizer.step()

                    if optimizer_was_run and self.deepspeed is None:
                        self.lr_scheduler.step()

                    model.zero_grad()
                    self.state.global_step += 1
                    self.state.epoch = epoch + (step + 1) / steps_in_epoch
                    self.control = self.callback_handler.on_step_end(args, self.state, self.control)

                    self._maybe_log_save_evaluate(tr_loss,mlm_loss, cls_loss, model, trial, epoch, ignore_keys_for_eval)
                else:
                    self.control = self.callback_handler.on_substep_end(args, self.state, self.control)

                if self.control.should_epoch_stop or self.control.should_training_stop:
                    break
            if step < 0:
                logger.warning(
                    "There seems to be not a single sample in your epoch_iterator, stopping training at step"
                    f" {self.state.global_step}! This is expected if you're using an IterableDataset and set"
                    f" num_steps ({max_steps}) higher than the number of available samples."
                )
                self.control.should_training_stop = True

            self.control = self.callback_handler.on_epoch_end(args, self.state, self.control)
            self._maybe_log_save_evaluate(tr_loss, mlm_loss, cls_loss, model, trial, epoch, ignore_keys_for_eval)

            if DebugOption.TPU_METRICS_DEBUG in self.args.debug:
                if is_torch_tpu_available():
                    # tpu-comment: Logging debug metrics for PyTorch/XLA (compile, execute times, ops, etc.)
                    xm.master_print(met.metrics_report())
                else:
                    logger.warning(
                        "You enabled PyTorch/XLA debug metrics but you don't have a TPU "
                        "configured. Check your training configuration if this is unexpected."
                    )
            if self.control.should_training_stop:
                break

        if args.past_index and hasattr(self, "_past"):
            # Clean the state at the end of training
            delattr(self, "_past")

        logger.info("\n\nTraining completed. Do not forget to share your model on huggingface.co/models =)\n\n")
        if args.load_best_model_at_end and self.state.best_model_checkpoint is not None:
            # Wait for everyone to get here so we are sur the model has been saved by process 0.
            if is_torch_tpu_available():
                xm.rendezvous("load_best_model_at_end")
            elif args.local_rank != -1:
                dist.barrier()
            elif is_sagemaker_mp_enabled():
                smp.barrier()

            self._load_best_model()

        # add remaining tr_loss
        self._total_loss_scalar += tr_loss.item()
        train_loss = self._total_loss_scalar / self.state.global_step
        
        # print(f"mlm_loss before total is: {mlm_loss.item()}")
        # same for mlm loss
        self._total_mlm_loss_scalar += mlm_loss.item()
        train_mlm_loss = self._total_mlm_loss_scalar / self.state.global_step
        # same for seq classifier loss           
        self._total_cls_loss_scalar += cls_loss.item()
        train_cls_loss = self._total_cls_loss_scalar / self.state.global_step
        
        metrics = speed_metrics("train", start_time, num_samples=num_train_samples, num_steps=self.state.max_steps)
        self.store_flos()
        metrics["total_flos"] = self.state.total_flos
        metrics["train_loss"] = train_loss
        #TODO add the separate mlm and seq loss here
        metrics["train_mlm_loss"] = train_mlm_loss
        metrics["train_seq_cls_loss"] = train_cls_loss
        self.is_in_train = False

        self._memory_tracker.stop_and_update_metrics(metrics)
        # print(f"metrics are: {metrics}")
        self.log(metrics)

        run_dir = self._get_output_dir(trial)
        checkpoints_sorted = self._sorted_checkpoints(use_mtime=False, output_dir=run_dir)

        # Delete the last checkpoint when save_total_limit=1 if it's different from the best checkpoint and process allowed to save.
        if self.args.should_save and self.state.best_model_checkpoint is not None and self.args.save_total_limit == 1:
            for checkpoint in checkpoints_sorted:
                if checkpoint != self.state.best_model_checkpoint:
                    logger.info(f"Deleting older checkpoint [{checkpoint}] due to args.save_total_limit")
                    shutil.rmtree(checkpoint)

        self.control = self.callback_handler.on_train_end(args, self.state, self.control)

        return TrainOutput(self.state.global_step, train_loss, metrics)
        
        
    def evaluation_loop(
            self,
            dataloader: DataLoader,
            description: str,
            prediction_loss_only: Optional[bool] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
        ) -> EvalLoopOutput:
            """
            Prediction/evaluation loop, shared by `Trainer.evaluate()` and `Trainer.predict()`.
            Works both with or without labels.
            """
            args = self.args

            prediction_loss_only = prediction_loss_only if prediction_loss_only is not None else args.prediction_loss_only

            # if eval is called w/o train init deepspeed here
            if args.deepspeed and self.deepspeed is None:

                # XXX: eval doesn't have `resume_from_checkpoint` arg but we should be able to do eval
                # from the checkpoint eventually
                deepspeed_engine, _, _ = deepspeed_init(
                    self, num_training_steps=0, resume_from_checkpoint=None, inference=True
                )
                self.model = deepspeed_engine.module
                self.model_wrapped = deepspeed_engine
                self.deepspeed = deepspeed_engine

            model = self._wrap_model(self.model, training=False, dataloader=dataloader)

            # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
            # while ``train`` is running, cast it to the right dtype first and then put on device
            if not self.is_in_train:
                if args.fp16_full_eval:
                    model = model.to(dtype=torch.float16, device=args.device)
                elif args.bf16_full_eval:
                    model = model.to(dtype=torch.bfloat16, device=args.device)

            batch_size = self.args.eval_batch_size

            logger.info(f"***** Running {description} *****")
            if has_length(dataloader):
                logger.info(f"  Num examples = {self.num_examples(dataloader)}")
            else:
                logger.info("  Num examples: Unknown")
            logger.info(f"  Batch size = {batch_size}")

            model.eval()

            self.callback_handler.eval_dataloader = dataloader
            # Do this before wrapping.
            eval_dataset = getattr(dataloader, "dataset", None)

            if is_torch_tpu_available():
                dataloader = pl.ParallelLoader(dataloader, [args.device]).per_device_loader(args.device)

            if args.past_index >= 0:
                self._past = None

            # Initialize containers
            # losses/preds/labels on GPU/TPU (accumulated for eval_accumulation_steps)
            losses_host = None
            preds_host = None
            labels_host = None
            inputs_host = None

            # losses/preds/labels on CPU (final containers)
            all_losses = None
            all_preds = None
            all_labels = None
            all_inputs = None
            # Will be useful when we have an iterable dataset so don't know its length.

            observed_num_examples = 0
            # Main evaluation loop
            for step, inputs in enumerate(dataloader):
                # Update the observed num examples
                observed_batch_size = find_batch_size(inputs)
                if observed_batch_size is not None:
                    observed_num_examples += observed_batch_size
                    # For batch samplers, batch_size is not known by the dataloader in advance.
                    if batch_size is None:
                        batch_size = observed_batch_size

                # Prediction step
                loss, logits, labels = self.prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)
                # print(f"logits inside eval loop: {logits}")
                # print(f"labels inside eval loop: {labels}")
                inputs_decode = self._prepare_input(inputs["input_ids"]) if args.include_inputs_for_metrics else None

                if is_torch_tpu_available():
                    xm.mark_step()

                # Update containers on host
                if loss is not None:
                    losses = self._nested_gather(loss.repeat(batch_size))
                    losses_host = losses if losses_host is None else torch.cat((losses_host, losses), dim=0)
                if labels is not None:
                    labels = self._pad_across_processes(labels)
                    labels = self._nested_gather(labels)
                    labels_host = labels if labels_host is None else nested_concat(labels_host, labels, padding_index=-100)
                if inputs_decode is not None:
                    inputs_decode = self._pad_across_processes(inputs_decode)
                    inputs_decode = self._nested_gather(inputs_decode)
                    inputs_host = (
                        inputs_decode
                        if inputs_host is None
                        else nested_concat(inputs_host, inputs_decode, padding_index=-100)
                    )
                if logits is not None:
                    logits = self._pad_across_processes(logits)
                    logits = self._nested_gather(logits)                    
                    if self.preprocess_logits_for_metrics is not None:
                        logits = self.preprocess_logits_for_metrics(logits, labels)
                    # if preds_host is None:
                    #     print(f"no preds host")
                    # else:
                    #     print(f"pred host just before potential concat: {preds_host}")
                        #TODO - at moment runs out of memory. So can try edit the preprocess_logits bit as that seems to resolve that issue somehow
                        # need to return both logits/preds for each task
                    preds_host = logits if preds_host is None else nested_concat(preds_host, logits, padding_index=-100)
                    # print(f"logits after setting to pred host: {preds_host}")
                self.control = self.callback_handler.on_prediction_step(args, self.state, self.control)

                # Gather all tensors and put them back on the CPU if we have done enough accumulation steps.
                if args.eval_accumulation_steps is not None and (step + 1) % args.eval_accumulation_steps == 0:
                    if losses_host is not None:
                        losses = nested_numpify(losses_host)
                        all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
                    if preds_host is not None:
                        logits = nested_numpify(preds_host)
                        all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                    if inputs_host is not None:
                        inputs_decode = nested_numpify(inputs_host)
                        all_inputs = (
                            inputs_decode
                            if all_inputs is None
                            else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                        )
                    if labels_host is not None:
                        labels = nested_numpify(labels_host)
                        all_labels = (
                            labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)
                        )

                    # Set back to None to begin a new accumulation
                    losses_host, preds_host, inputs_host, labels_host = None, None, None, None

            if args.past_index and hasattr(self, "_past"):
                # Clean the state at the end of the evaluation loop
                delattr(self, "_past")

            # Gather all remaining tensors and put them back on the CPU
            if losses_host is not None:
                losses = nested_numpify(losses_host)
                all_losses = losses if all_losses is None else np.concatenate((all_losses, losses), axis=0)
            if preds_host is not None:
                # print(f"we have preds host: {preds_host}")
                logits = nested_numpify(preds_host)
                # print(f"logits after nested numpify: {logits}")
                all_preds = logits if all_preds is None else nested_concat(all_preds, logits, padding_index=-100)
                # print(f"all_preds now: {all_preds}")
            if inputs_host is not None:
                inputs_decode = nested_numpify(inputs_host)
                all_inputs = (
                    inputs_decode if all_inputs is None else nested_concat(all_inputs, inputs_decode, padding_index=-100)
                )
            if labels_host is not None:
                labels = nested_numpify(labels_host)
                all_labels = labels if all_labels is None else nested_concat(all_labels, labels, padding_index=-100)

            # Number of samples
            if has_length(eval_dataset):
                num_samples = len(eval_dataset)
            # The instance check is weird and does not actually check for the type, but whether the dataset has the right
            # methods. Therefore we need to make sure it also has the attribute.
            elif isinstance(eval_dataset, IterableDatasetShard) and getattr(eval_dataset, "num_examples", 0) > 0:
                num_samples = eval_dataset.num_examples
            else:
                if has_length(dataloader):
                    num_samples = self.num_examples(dataloader)
                else:  # both len(dataloader.dataset) and len(dataloader) fail
                    num_samples = observed_num_examples
            if num_samples == 0 and observed_num_examples > 0:
                num_samples = observed_num_examples

            # Number of losses has been rounded to a multiple of batch_size and in a distributed training, the number of
            # samplers has been rounded to a multiple of batch_size, so we truncate.
            if all_losses is not None:
                all_losses = all_losses[:num_samples]
            if all_preds is not None:
                all_preds = nested_truncate(all_preds, num_samples)
            if all_labels is not None:
                all_labels = nested_truncate(all_labels, num_samples)
            if all_inputs is not None:
                all_inputs = nested_truncate(all_inputs, num_samples)

            # Metrics!
            
            #TODO edit this to allow calc of f1 etc for classification task
            if self.compute_metrics is not None and all_preds is not None and all_labels is not None:
                if args.include_inputs_for_metrics:
                    metrics = self.compute_metrics(
                        EvalPrediction(predictions=all_preds, label_ids=all_labels, inputs=all_inputs)
                    )
                else:
                    #TODO - this may need editing to allow the separate label ids for each task
                    metrics = self.compute_metrics(EvalPrediction(predictions=all_preds, label_ids=all_labels))
            else:
                metrics = {}

            # To be JSON-serializable, we need to remove numpy types or zero-d tensors
            metrics = denumpify_detensorize(metrics)

            if all_losses is not None:
                metrics[f"{metric_key_prefix}_loss"] = all_losses.mean().item()
            if hasattr(self, "jit_compilation_time"):
                metrics[f"{metric_key_prefix}_jit_compilation_time"] = self.jit_compilation_time

            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            return EvalLoopOutput(predictions=all_preds, label_ids=all_labels, metrics=metrics, num_samples=num_samples)
        
    def prediction_step(
            self,
            model: nn.Module,
            inputs: Dict[str, Union[torch.Tensor, Any]],
            prediction_loss_only: bool,
            ignore_keys: Optional[List[str]] = None,
        ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
            """
            Perform an evaluation step on `model` using `inputs`.
            Subclass and override to inject custom behavior.
            Args:
                model (`nn.Module`):
                    The model to evaluate.
                inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                    The inputs and targets of the model.
                    The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                    argument `labels`. Check your model's documentation for all accepted arguments.
                prediction_loss_only (`bool`):
                    Whether or not to return the loss only.
                ignore_keys (`Lst[str]`, *optional*):
                    A list of keys in the output of your model (if it is a dictionary) that should be ignored when
                    gathering predictions.
            Return:
                Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple with the loss,
                logits and labels (each being optional).
            """
            # print(f"self.label_names are: {self.label_names}")
            
            has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
            
            # For CLIP-like models capable of returning loss values.
            # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
            # is `True` in `model.forward`.
            return_loss = inputs.get("return_loss", None)
            if return_loss is None:
                return_loss = self.can_return_loss
            loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

            inputs = self._prepare_inputs(inputs)
            if ignore_keys is None:
                if hasattr(self.model, "config"):
                    ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
                else:
                    ignore_keys = []

            # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
            if has_labels or loss_without_labels:
                # print(f"label names: {self.label_names}")
                labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
                if len(labels) == 1:
                    labels = labels[0]
            else:
                labels = None

            with torch.no_grad():
                # RIGHT NOW IGNORING SAGEMAKER ENTIRELY
                if is_sagemaker_mp_enabled():
                    raw_outputs = smp_forward_only(model, inputs)
                    if has_labels or loss_without_labels:
                        if isinstance(raw_outputs, dict):
                            loss_mb = raw_outputs["loss"]
                            logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys + ["loss"])
                        else:
                            loss_mb = raw_outputs[0]
                            logits_mb = raw_outputs[1:]

                        loss = loss_mb.reduce_mean().detach().cpu()
                        logits = smp_nested_concat(logits_mb)
                    else:
                        loss = None
                        if isinstance(raw_outputs, dict):
                            logits_mb = tuple(v for k, v in raw_outputs.items() if k not in ignore_keys)
                        else:
                            logits_mb = raw_outputs
                        logits = smp_nested_concat(logits_mb)
                else:
                    if has_labels or loss_without_labels:
                        with self.compute_loss_context_manager():
                            #TODO - add handling of contrastive loss here
                            loss, mlm_loss, cls_loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                        loss = loss.mean().detach()
                        mlm_loss = mlm_loss.mean().detach()
                        if cls_loss is not None:
                            cls_loss = cls_loss.mean().detach()
                        if isinstance(outputs, dict):
                            # print(f"outputs are: {outputs}")
                            if outputs.seq_classifier_logits is None:                            
                            #### original - is annoying without ignore keys - we know we only want prediction_logits so will get them explicitly
                                # logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss","mlm_loss", "cls_loss", "seq_classifier_logits"])
                                logits = tuple([outputs.prediction_logits])
                            ####
                            else:
                                # print(f" we got seq classifier logits?")
                                # get logits for both tasks
                                logits = tuple([outputs.prediction_logits, outputs.seq_classifier_logits])
                            # print(f"logits inside pred step are: {logits}\n\n")
                            # print(f"new logits are: {new_logits}")
                            # print(f"length of logits: {len(logits)}")
                        else:
                            # change this to separate logits for 
                            logits = outputs[1:]
                    else:
                        loss = None
                        with self.compute_loss_context_manager():
                            outputs = model(**inputs)
                        if isinstance(outputs, dict):
                            logits = tuple(v for k, v in outputs.items() if k not in ignore_keys)
                        else:
                            logits = outputs
                        # TODO: this needs to be fixed and made cleaner later.
                        if self.args.past_index >= 0:
                            self._past = outputs[self.args.past_index - 1]
            #TODO also return mlm_loss and cls_loss etc
            if prediction_loss_only:
                
                return (loss, None, None)
            # print(f"logits before nested detach are: {logits}")
            logits = nested_detach(logits)
            if len(logits) == 1:
                logits = logits[0]
            # print(f"logits being returned: {logits}")
            return (loss, logits, labels)
        
    def _maybe_log_save_evaluate(self, tr_loss, mlm_loss, cls_loss, model, trial, epoch, ignore_keys_for_eval):
        # print(f"inside maybe save log")
        if self.control.should_log:
            # print(f"we will be logging")
            if is_torch_tpu_available():
                xm.mark_step()

            logs: Dict[str, float] = {}

            # all_gather + mean() to get average loss over all processes
            tr_loss_scalar = self._nested_gather(tr_loss).mean().item()
            if mlm_loss is not None:
                # print(f"updated tr_mlm_scalar")
                tr_mlm_scalar = self._nested_gather(mlm_loss).mean().item()
            if cls_loss is not None:
                # print(f"updating tr_cls_scalar")
                tr_cls_scalar = self._nested_gather(cls_loss).mean().item()

            # reset tr_loss to zero
            tr_loss -= tr_loss
            if mlm_loss is not None:
                mlm_loss -= mlm_loss
            if cls_loss is not None:
                cls_loss -= cls_loss            
            
            
            logs["loss"] = round(tr_loss_scalar / (self.state.global_step - self._globalstep_last_logged), 4)            
            logs["learning_rate"] = self._get_learning_rate()
            
            if mlm_loss is not None:
                # print(f"got mlm_loss and will add to logs here")
                logs["mlm_loss"] = round(tr_mlm_scalar / (self.state.global_step - self._globalstep_last_logged), 4)  
                self._total_mlm_loss_scalar += tr_mlm_scalar
                
            if cls_loss is not None:
                # print(f"got cls loss and will add to logs here")
                logs["cls_loss"] = round(tr_cls_scalar / (self.state.global_step - self._globalstep_last_logged), 4)  
                self._total_cls_loss_scalar += tr_cls_scalar
            self._total_loss_scalar += tr_loss_scalar
            self._globalstep_last_logged = self.state.global_step
            self.store_flos()          
            

            self.log(logs)

        metrics = None
        if self.control.should_evaluate:
            if isinstance(self.eval_dataset, dict):
                for eval_dataset_name, eval_dataset in self.eval_dataset.items():
                    metrics = self.evaluate(
                        eval_dataset=eval_dataset,
                        ignore_keys=ignore_keys_for_eval,
                        metric_key_prefix=f"eval_{eval_dataset_name}",
                    )
            else:
                metrics = self.evaluate(ignore_keys=ignore_keys_for_eval)
            self._report_to_hp_search(trial, self.state.global_step, metrics)

        if self.control.should_save:
            self._save_checkpoint(model, trial, metrics=metrics)
            self.control = self.callback_handler.on_save(self.args, self.state, self.control)
            
class PreTrainingEvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.
    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        mlm_predictions: Union[np.ndarray, Tuple[np.ndarray]],
        mlm_label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        cls_predictions: Union[np.ndarray, Tuple[np.ndarray]],
        cls_label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.mlm_predictions = mlm_predictions
        self.mlm_label_ids = mlm_label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs
        
class PreTrainingEvalLoopOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]
    num_samples: Optional[int]