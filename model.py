from transformers import AutoModel, AutoConfig
import torch.nn as nn
import torch
import torch.functional as F
import numpy as np

class AttentionModel(nn.Module):
  def __init__(self, checkpoint_base, checkpoint_sentiment, num_labels, freeze_base=True, device='cuda'): 
    super(AttentionModel,self).__init__() 
    self.num_labels = num_labels 

    #Load Model with given checkpoint and extract its body
    base_config = AutoConfig.from_pretrained(checkpoint_base, output_attentions=False,output_hidden_states=False)
    self.model_base = AutoModel.from_pretrained(checkpoint_base,config=base_config)
    sentiment_config = AutoConfig.from_pretrained(checkpoint_sentiment, output_attentions=False,output_hidden_states=False)
    self.model_sentiment = AutoModel.from_pretrained(checkpoint_sentiment,config=sentiment_config)

    # Attention Layer to combine both outputs
    self.attn_layer = AttentionLayer()
    # Fully Connected Layer for classification
    self.ffn_layer = nn.Linear(768, 1)
    # Loss function
    self.loss_fn = nn.BCEWithLogitsLoss()
    
    if freeze_base:
      for name, param in self.model_base.named_parameters():
          param.requires_grad = False
      for name, param in self.model_sentiment.named_parameters():
          param.requires_grad = False
    

  def forward(self, input_ids=None, attention_mask=None, labels=None, loss=True):
    #Extract outputs from the body
    outputs_base = self.model_base(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
    outputs_sentiment = self.model_sentiment(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
    
    x = self.attn_layer(outputs_base, outputs_sentiment, attention_mask)
    x = self.ffn_layer(x)
    if loss:
      x = self.loss_fn(x, labels.unsqueeze(1).float())
    
    return x
  

class AttentionLayer(nn.Module):
    def __init__(self, token_dim=768, cross_dim=768):
        super(AttentionLayer, self).__init__()
        self.cross_token_param = nn.Linear(token_dim * 2, cross_dim)
        self.ind_token_param = nn.Linear(token_dim * 2, 1)
        self.att_token_param = torch.nn.Parameter(torch.FloatTensor(cross_dim, 1), requires_grad=True)
        torch.nn.init.xavier_uniform(self.att_token_param)

    def masked_softmax(self, x, mask, dim=1):
      x_masked = x.clone()
      x_masked[mask == 0] = -float("inf")

      return torch.softmax(x_masked, dim=dim)


    def forward(self, outputs_base, outputs_sentiment, attention_mask=None):
        # Concatenate the outputs of both models
        hidden_concat = torch.cat((outputs_base, outputs_sentiment), dim=2)
        # Apply cross-token attention
        cross_token = torch.tanh(self.cross_token_param(hidden_concat))
        # Apply individual token attention
        ind_token = torch.sigmoid(self.ind_token_param(hidden_concat))
        # Apply attention weights
        att_token = torch.squeeze(torch.matmul(cross_token, self.att_token_param))
        # Apply softmax to get attention weights and apply attention mask
        att_token = self.masked_softmax(att_token, attention_mask, 1)
        # Expand attention weights to match the hidden states dimension
        att_token = att_token.unsqueeze(2)
        att_token = att_token.expand(-1, -1, 768)
        # Get the inter token representation
        inter_token = (1 - ind_token) * outputs_base + ind_token * outputs_sentiment
        # Apply attention weights to the hidden states
        final_representation = torch.mul(att_token, inter_token).sum(dim=1)

        return final_representation



class AttentionModelSingle(nn.Module):
  def __init__(self, checkpoint_base, num_labels, freeze_base=True, device='cuda'): 
    super(AttentionModelSingle,self).__init__() 
    self.num_labels = num_labels 

    #Load Model with given checkpoint and extract its body
    base_config = AutoConfig.from_pretrained(checkpoint_base, output_attentions=False,output_hidden_states=False)
    self.model_base = AutoModel.from_pretrained(checkpoint_base,config=base_config)

    # Attention Layer to combine both outputs
    self.attn_layer = AttentionLayerSingle()
    # Fully Connected Layer for classification
    self.ffn_layer = nn.Linear(768, 1)
    # Loss function
    self.loss_fn = nn.BCEWithLogitsLoss()
    
    if freeze_base:
      for name, param in self.model_base.named_parameters():
          param.requires_grad = False

  def forward(self, input_ids=None, attention_mask=None, labels=None, loss=True):
    #Extract outputs from the body
    outputs_base = self.model_base(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state']
    
    x = self.attn_layer(outputs_base, attention_mask)
    x = self.ffn_layer(x)
    if loss:
      x = self.loss_fn(x, labels.unsqueeze(1).float())
    
    return x
  
class AttentionLayerSingle(nn.Module):
  def __init__(self, token_dim=768, cross_dim=768):
      super(AttentionLayerSingle, self).__init__()
      self.cross_token_param = nn.Linear(token_dim, cross_dim)
      self.att_token_param = torch.nn.Parameter(torch.FloatTensor(cross_dim, 1), requires_grad=True)
      torch.nn.init.xavier_uniform(self.att_token_param)

  def masked_softmax(self, x, mask, dim=1):
    x_masked = x.clone()
    x_masked[mask == 0] = -float("inf")

    return torch.softmax(x_masked, dim=dim)


  def forward(self, outputs_base, attention_mask=None):
      # Apply cross-token attention
      cross_token = torch.tanh(self.cross_token_param(outputs_base))
      # Apply attention weights
      att_token = torch.squeeze(torch.matmul(cross_token, self.att_token_param))
      # Apply softmax to get attention weights and apply attention mask
      att_token = self.masked_softmax(att_token, attention_mask, 1)
      # Expand attention weights to match the hidden states dimension
      att_token = att_token.unsqueeze(2)
      att_token = att_token.expand(-1, -1, 768)
      # Apply attention weights to the hidden states
      final_representation = torch.mul(att_token, outputs_base).sum(dim=1)

      return final_representation
  

class TwoCLSModel(nn.Module):
  def __init__(self, checkpoint_base, checkpoint_sentiment, num_labels, freeze_base=True, device='cuda'): 
    super(TwoCLSModel,self).__init__() 
    self.num_labels = num_labels 

    #Load Model with given checkpoint and extract its body
    base_config = AutoConfig.from_pretrained(checkpoint_base, output_attentions=False,output_hidden_states=False)
    self.model_base = AutoModel.from_pretrained(checkpoint_base,config=base_config)
    sentiment_config = AutoConfig.from_pretrained(checkpoint_sentiment, output_attentions=False,output_hidden_states=False)
    self.model_sentiment = AutoModel.from_pretrained(checkpoint_sentiment,config=sentiment_config)

    # Fully Connected Layer for classification
    self.ffn_layer = nn.Linear(2*768, 768)
    self.ffn_layer2 = nn.Linear(768, 1)
    # Loss function
    self.loss_fn = nn.BCEWithLogitsLoss()
    
    if freeze_base:
      for name, param in self.model_base.named_parameters():
          param.requires_grad = False
      for name, param in self.model_sentiment.named_parameters():
          param.requires_grad = False

  def forward(self, input_ids=None, attention_mask=None, labels=None, loss=True):
    #Extract outputs from the body
    outputs_base = self.model_base(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:,0,:] #CLS token
    outputs_sentiment = self.model_sentiment(input_ids=input_ids, attention_mask=attention_mask)['last_hidden_state'][:,0,:] #CLS token

    # Concatenate the outputs of both models
    x = torch.cat((outputs_base, outputs_sentiment), dim=1)
    
    x = torch.sigmoid(self.ffn_layer(x))
    x = self.ffn_layer2(x)
    if loss:
      x = self.loss_fn(x, labels.unsqueeze(1).float())
    
    return x