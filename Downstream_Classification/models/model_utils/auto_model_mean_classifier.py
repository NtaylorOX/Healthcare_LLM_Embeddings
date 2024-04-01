from transformers import (
    PreTrainedModel,
    BertPreTrainedModel, 
    BertModel, 
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
)



import math
from typing import List, Optional, Tuple, Union

import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss


class MeanSequenceClassificationHead(nn.Module):
    """Custom Head for sentence-level classification tasks. A slight tweak to the original implementation from 
        the transformers/orignal authors found here: https://github.com/huggingface/transformers/blob/main/src/transformers/models/roberta/modeling_roberta.py
        
        Here we use the averaged token embeddings from the final transformer layer - this is similar to other work such as
        DeCLUTR and SentenceTransformers which look to produce sentence level embeddings via contrastive loss functions etc.

        This is a preferred sentence classification model if you are building upon any pre-training which has utilised 
        averaged embeddings before. 
        
        """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        # x here by default is all of the last hidden states of shape [L x S x E] where B is batch_size, S is sequence length and E is embedding dimension
        # instead we altered the forward pass of the model to supply the single embedding directly
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])         
        
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class MeanBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        
        self.classifier = MeanBertClassificationHead(config)
        # Initialize weights and apply final processing
        self.post_init()


    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
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

        sequence_output = torch.sum(
            outputs[0] * attention_mask.unsqueeze(-1), dim=1
        ) / torch.clamp(torch.sum(attention_mask, dim=1, keepdims=True), min=1e-9)
        
        # print(f"Sequence shape is now: {sequence_output.shape}")
        # # obtaining the last layer hidden states of the Transformer
        # last_hidden_state = outputs.last_hidden_state  # shape: (batch_size, seq_length, bert_hidden_dim)

        # #         or can use the output pooler : output = self.classifier(output.pooler_output)
        # # As I said, the CLS token is in the beginning of the sequence. So, we grab its representation
        # # by indexing the tensor containing the hidden representations
        # CLS_token_state = last_hidden_state[:, 0, :]
        # passing this representation through our custom head
        
        # original robert class just takes the 0th element    
        # sequence_output = outputs[0]
        
        logits = self.classifier(sequence_output)

        loss = None
        if labels is not None:
            if self.config.problem_type is None:
                if self.num_labels == 1:
                    self.config.problem_type = "regression"
                elif self.num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.config.problem_type = "single_label_classification"
                else:
                    self.config.problem_type = "multi_label_classification"

            if self.config.problem_type == "regression":
                loss_fct = MSELoss()
                if self.num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif self.config.problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            elif self.config.problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss()
                loss = loss_fct(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )