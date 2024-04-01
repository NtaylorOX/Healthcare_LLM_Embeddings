import torch
import torch.nn as nn


#### Embedding quality loss functions
''' 
The two functions below: align_loss and uniform_loss are taken from: https://arxiv.org/pdf/2104.08821.pdf 


'''
def align_loss(x, y, alpha=2):
    '''
    Given a distribution of positive
    pairs ppos, alignment calculates expected distance
    between embeddings of the paired instances (as-
    suming representations are already normalized)/
        
    args:
        x: tensor of shape (batch_size, embedding_dim) -> original
        y: tensor of shape (batch_size, embedding_dim) -> positive pair 
    '''
    
    return (x - y).norm(p=2, dim=1).pow(alpha).mean()


def uniform_loss(x, t=2):
    
    '''  uniformity measures how well
    the embeddings are uniformly distributed:''' 
    return torch.pdist(x, p=2).pow(2).mul(-t).exp().mean().log()


def freeze_n_layers(model, freeze_layer_count = 0) -> None:
    """unfreeze N last layers of a transformer model"""
    # first freeze the embedding layer
    #TODO - update to work for other PLMs like gpt which do not have embeddings attribute
    for param in model.base_model.embeddings.parameters():
            param.requires_grad = False
    # if the freeze layer count is 0 - do nothing and leave requires_grad = True i.e. the default after loading model in
    
    if freeze_layer_count > model.config.num_hidden_layers:
        print(f"The freeze_layer_count provided:{freeze_layer_count} is higher than the number of layers the model has: {model.config.num_hidden_layers}!  ")
    else:
        if freeze_layer_count != 0:        
        
            if freeze_layer_count != -1:
                # if freeze_layer_count == -1, we freeze all of em
                # otherwise we freeze the first `freeze_layer_count` encoder layers
                for layer in model.base_model.encoder.layer[:freeze_layer_count]:
                    for param in layer.parameters():
                        param.requires_grad = False
            else:
                # TODO - currently this is not working as expected for following models: /mnt/sdc/niallt/saved_models/language_modelling/mimic/roberta-base-mimic-wecho/sampled_250000/08-03-2023--13-06/checkpoint-84000/
                for layer in model.base_model.encoder.layer:
                    for param in layer.parameters():
                        param.requires_grad = False
                    

def unfreeze_encoder(model) -> None:
    """ un-freezes the encoder layer. """    
    
    for param in model.parameters():
        if not param.requires_grad:
            param.requires_grad = True
        # the name of the PLM component depends on the architecture/pretrained model
        # if "roberta" in self.model.name_or_path:                 
            
        #     for param in self.model.roberta.parameters(): # can maybe replace with self.model.base_model? and avoid this if roberta or bert business?
        #         param.requires_grad = True
                
        # elif "bert" in self.model.name_or_path:
        #     for param in self.model.bert.parameters():
        #         param.requires_grad = True

                
        # else:
        #     raise NotImplementedError


def freeze_encoder(model) -> None:
    """ freezes the encoder layer. """
    
    for param in model.base_model.parameters():
        param.requires_grad = False
        # the name of the PLM component depends on the architecture/pretrained model
        # if "roberta" in self.model.name_or_path:                 
            
        #     for param in self.model.roberta.parameters():
        #         param.requires_grad = False
                
        # elif "bert" in self.model.name_or_path:
        #     for param in self.model.bert.parameters():
        #         param.requires_grad = False
        # else:
        #     raise NotImplementedError
        
def count_trainable_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)