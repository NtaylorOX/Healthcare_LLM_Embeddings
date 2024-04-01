from transformers import ( # currently working with transformers 4.27.3
    PreTrainedModel,
    RobertaPreTrainedModel, 
    RobertaModel, 
    BertModel,
    BertPreTrainedModel,
    PretrainedConfig, 
    RobertaConfig, 
    RobertaForSequenceClassification,
    AutoModelForSequenceClassification,
    AutoConfig, 
    AutoModel,
    RobertaPreTrainedModel
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
    ModelOutput
)

from transformers.activations import ACT2FN, gelu

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from pytorch_metric_learning import losses
# loss_func = losses.TripletMarginLoss()
# loss_func = losses.SupConLoss(temperature=0.1)
# loss_func = losses.NTXentLoss(temperature=0.1)

#NOTE this will only work cleanly for models based on RoBERTA - this is due to vocab sizes and config causing different embedding/vocab sizes etc

class MeanRobertaConfig(RobertaConfig):
    r"""
    This is the configuration class to store the configuration of a :class:`~transformers.RobertaModel` or a
    :class:`~transformers.TFRobertaModel`. It is used to instantiate a RoBERTa model according to the specified
    arguments, defining the model architecture.


    Configuration objects inherit from :class:`~transformers.PretrainedConfig` and can be used to control the model
    outputs. Read the documentation from :class:`~transformers.PretrainedConfig` for more information.

    The :class:`~transformers.RobertaConfig` class directly inherits :class:`~transformers.BertConfig`. It reuses the
    same defaults. Please check the parent class for more information.

    Examples::

        >>> from transformers import RobertaConfig, RobertaModel

        >>> # Initializing a RoBERTa configuration
        >>> configuration = RobertaConfig()

        >>> # Initializing a model from the configuration
        >>> model = RobertaModel(configuration)

        >>> # Accessing the model configuration
        >>> configuration = model.config
    """
    model_type = "roberta" # if using MeanRobertaConfig and wanting to register as new model - use "meanroberta"

    def __init__(self, pad_token_id=1,
                 bos_token_id=0,
                 eos_token_id=2,
                 compute_contrastive = None,
                 num_pretraining_labels = 12,
                 contrastive_loss_weight = 1.0,
                 compute_note_loss_only = False,
                 compute_mlm_loss_only = False,
                 **kwargs):
        """Constructs RobertaConfig."""
        # add some extra arguments
        self.compute_contrastive = compute_contrastive
        self.num_pretraining_labels = num_pretraining_labels
        self.contrastive_loss_weight = contrastive_loss_weight
        self.compute_note_loss_only = compute_note_loss_only
        self.compute_mlm_loss_only = compute_mlm_loss_only
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)
        

        
class TransformerPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class TransformerLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = TransformerPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

class TransformerPreTrainingHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        
        self.lm_head = TransformerLMPredictionHead(config)
        self.seq_classifier = nn.Linear(config.hidden_size, config.num_pretraining_labels) # need to edit this to take a value from config or something

    def forward(self, sequence_output, pooled_output):
        prediction_scores = self.lm_head(sequence_output)
        # print(f"pooled output here: {pooled_output} with shape: {pooled_output.shape}")
        seq_classifier_score = self.seq_classifier(pooled_output)
        return prediction_scores, seq_classifier_score
    
class MeanSequenceClassifier(nn.Module):
    def __init__(self, config):
        super().__init__()       

        self.classifier = nn.Linear(config.hidden_size, config.num_pretraining_labels) # need to edit this to take a value from config or something

    def forward(self, pooled_output):
        # print(f"pooled output here: {pooled_output} with shape: {pooled_output.shape}")
        seq_classifier_score = self.classifier(pooled_output)
        return seq_classifier_score
    

class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = gelu(x)
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        # For accelerate compatibility and to not break backward compatibility
        if self.decoder.bias.device.type == "meta":
            self.decoder.bias = self.bias
        else:
            self.bias = self.decoder.bias

class TransformerForPreTrainingOutput(ModelOutput):
    """
    Output type of [`BertForPreTraining`].
    Args:
        loss (*optional*, returned when `labels` is provided, `torch.FloatTensor` of shape `(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_classifier_logits (`torch.FloatTensor` of shape `(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer) of
            shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None # total loss
    mlm_loss:Optional[torch.FloatTensor] = None
    cls_loss: Optional[torch.FloatTensor] = None
    prediction_logits: torch.FloatTensor = None
    seq_classifier_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


    
    
class TransformerForPreTraining(RobertaPreTrainedModel):
    
    # add custom config class
    config_class = MeanRobertaConfig
    # def __init__(self, config):
    #     super().__init__(config)

    #     self.bert = BertModel(config)
    #     self.cls = TransformerPreTrainingHeads(config)



    #     # Initialize weights and apply final processing
    #     self.post_init()

    # def get_output_embeddings(self):
    #     return self.cls.predictions.decoder

    # def set_output_embeddings(self, new_embeddings):
    #     self.cls.predictions.decoder = new_embeddings 

    def __init__(self, config):
        super().__init__(config)

        if config.is_decoder:
            logger.warning(
                "If you want to use `RobertaForMaskedLM` make sure `config.is_decoder=False` for "
                "bi-directional self-attention."
            )
        #TODO - add PEFT
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.lm_head = RobertaLMHead(config)
        self.seq_classifier = MeanSequenceClassifier(config)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        # Initialize weights and apply final processing
        self.post_init()
        
        # here compute contrastive just refers to using the contrastive loss function rather than sequence classification
        if config.compute_contrastive:            
            self.compute_contrastive = True

 
        else:
            self.compute_contrastive = False

            
        # are we only computing note_category loss
        self.compute_note_loss_only = config.compute_note_loss_only
        
        # are we only computing mlm loss
        self.compute_mlm_loss_only = config.compute_mlm_loss_only
        
        # assert that these two are not both true
        assert not (self.compute_note_loss_only and self.compute_mlm_loss_only), "Both compute_note_loss_only and compute_mlm_loss_only cannot be true at the same time"
        
        # add contrastive loss
        self.contrastive_loss_weight = config.contrastive_loss_weight
            
        # set number of catgeory class labels
        self.num_pretraining_labels = config.num_pretraining_labels
        
        

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        category_label: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        # compute_contrastive: Optional[bool] = None, # i think this is not needed as handeled by init        
    ) -> Union[Tuple[torch.Tensor], TransformerForPreTrainingOutput]:
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            category_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:
                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.
        Returns:
        Example:
        ```python
        >>> from transformers import AutoTokenizer, BertForPreTraining
        >>> import torch
        >>> tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        >>> model = BertForPreTraining.from_pretrained("bert-base-uncased")
        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)
        >>> prediction_logits = outputs.prediction_logits
        >>> seq_classifier_logits = outputs.seq_classifier_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # edit this to output mean hidden layer embeds
        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )               
        # print(f"outputs:\n {outputs}")
        # change this to take the mean of last hidden states as the "pooled output"
        sequence_output = outputs[0]
        pooled_output  = torch.sum(
            outputs[0] * attention_mask.unsqueeze(-1), dim=1
        ) / torch.clamp(torch.sum(attention_mask, dim=1, keepdims=True), min=1e-9)
        
               
        # print(f"prior to being sent to the classifier the shape of sequence output is: {sequence_output.shape}, and pooled output is: {pooled_output.shape}")
        # 
        # prediction_scores, seq_classifier_score = self.cls(sequence_output, pooled_output)
        prediction_scores = self.lm_head(sequence_output)
        
        #TODO - un comment this when using both losses
        # contrastive here refers to the loss function - i.e. contrastive loss or sequence classification loss
        if not self.compute_contrastive:
            seq_classifier_score = self.seq_classifier(pooled_output)
        else:
            seq_classifier_score = None
        #################      
               
        total_loss = None
        masked_lm_loss = None
        seq_classification_loss = None  
        # if both mlm labels and category label is provided                   
        if labels is not None and category_label is not None:                      
            loss_fct = CrossEntropyLoss()
            
            # check if we want MLM loss at all
            if not self.compute_note_loss_only:                
                masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            else:
                # set to 0 if we are only computing note loss
                masked_lm_loss = torch.tensor(0.0)
                
            # now check if we want just mlm only
            
            if not self.compute_mlm_loss_only:
            
                # print(f"masked lm loss: {masked_lm_loss}")
                # print(f"seq classifier shape: {seq_classifier_score.shape} and category label: {category_label}")
                # print(f"seq_classification_score: {seq_classifier_score} with shape: {seq_classifier_score.shape} and labels: {category_label} with shape: {category_label.shape}")
                #### correct way to get seq_classification loss
            
            
            
                if not self.compute_contrastive:                            
                    seq_classification_loss = loss_fct(seq_classifier_score.view(-1, self.num_pretraining_labels), category_label.view(-1))
                
                    #TODO add weighting to contrastive loss
                    #### combined losses            
                    total_loss = masked_lm_loss + self.contrastive_loss_weight * seq_classification_loss
                else:
                    contrastive_loss_fn = losses.SupConLoss(temperature=0.1)
                    # print(f"seq embeddings i.e. pooled output: {pooled_output} \n\n and category labels: {category_label}")
                    seq_classification_loss = contrastive_loss_fn(pooled_output, category_label)
                    # print(f"will be computing contrastive loss")
                    # print(f"pooled output shape: {pooled_output.shape}")
                    # just compute masked lm loss here -  will combine later with contrastive loss
                    #TODO add weighting to contrastive loss
                    total_loss = masked_lm_loss + self.contrastive_loss_weight * seq_classification_loss
                
            # if we are just computing mlm loss
            else:
                total_loss = masked_lm_loss
            
            
            
            ####### CHANKY FORCE FOR MLM loss only #####
            #### to return mlm loss only am manually setting seq_classification_loss to None - otherwise torch.distributed panics as it has gradients not used or something           
            
            #TODO - test returning only the mlm loss
            # seq_classification_loss = None
            
            
            # CRUDE FIX - just return masked_lm loss here
            # total_loss = masked_lm_loss
        #FIXME - right now this only works for classification head - need to add option for contrastive loss
        
        
        elif category_label is not None and not self.compute_contrastive:
            
            loss_fct = CrossEntropyLoss()
            seq_classification_loss = loss_fct(seq_classifier_score.view(-1, self.num_pretraining_labels), category_label.view(-1))
            total_loss = seq_classification_loss
        elif category_label is not None and self.compute_contrastive:
                contrastive_loss_fn = losses.SupConLoss(temperature=0.1)
                # print(f"seq embeddings i.e. pooled output: {pooled_output} \n\n and category labels: {category_label}")
                seq_classification_loss = contrastive_loss_fn(pooled_output, category_label)
                total_loss = seq_classification_loss
        if not return_dict:
            output = (prediction_scores, seq_classifier_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return TransformerForPreTrainingOutput(
            loss=total_loss,
            mlm_loss=masked_lm_loss,
            cls_loss=seq_classification_loss,
            prediction_logits=prediction_scores,
            seq_classifier_logits=seq_classifier_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            seq_embedding = pooled_output
        )