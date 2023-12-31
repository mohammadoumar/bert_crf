# We define the 2 classes BERT_DistilBERT_CRF and DistilBERT_CRF

from torchcrf import CRF

import torch.nn as nn
import torch

from transformers import BertModel, BertPreTrainedModel
from transformers import DistilBertModel, DistilBertPreTrainedModel


class BERT_CRF(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.crf = CRF(num_tags=config.num_labels, batch_first=True) # *** MODIF ***
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.view(-1) == 1
#                 active_logits = logits.view(-1, self.num_labels)
#                 active_labels = torch.where(
#                     active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
#                 )
#                 loss = loss_fct(active_logits, active_labels)
                
#             else:
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), scores, (hidden_states), (attentions)

        if labels is not None:
            labels_crf = labels.clone()      # copy of labels used for crf loss
            labels_crf[labels_crf==-100] = 0 # remove -100 otherwise not working!
            attention_mask = attention_mask.byte()
            
            loss = -self.crf(logits, labels_crf, mask=attention_mask) # CRF loss is positive
            
            # NOTE: self.crf.decode(logits, mask=attention_mask) is a list of lists
            # In order to use the trainer.predict() fct, convert it into a PADDED TENSOR! 
            max_size = labels_crf.shape[1]
            predictions_ll = self.crf.decode(logits, mask=attention_mask)            # list of lists
            predictions_ll = [ l + [-100]*(max_size-len(l)) for l in predictions_ll] # padded list of lists
            predictions_t = torch.Tensor(predictions_ll).int()                       # tensor
    
            outputs = (loss, predictions_t)

        return outputs  # (loss), scores, (hidden_states), (attentions)



class DistilBERT_CRF(DistilBertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.distilbert = DistilBertModel(config)
        self.dropout = nn.Dropout(config.dropout) # *** MODIF ***
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
        self.crf = CRF(num_tags=config.num_labels, batch_first=True) # *** MODIF ***
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        # token_type_ids=None, # *** MODIF ***
        # position_ids=None,   # *** MODIF ***
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Labels for computing the token classification loss.
            Indices should be in ``[0, ..., config.num_labels - 1]``.

    Returns:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        loss (:obj:`torch.FloatTensor` of shape :obj:`(1,)`, `optional`, returned when ``labels`` is provided) :
            Classification loss.
        scores (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.num_labels)`)
            Classification scores (before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        """

        outputs = self.distilbert(
            input_ids,
            attention_mask=attention_mask,
            # token_type_ids=token_type_ids, # *** MODIF ***
            # position_ids=position_ids,     # *** MODIF ***
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        
#         if labels is not None:
#             loss_fct = CrossEntropyLoss()
#             # Only keep active parts of the loss
#             if attention_mask is not None:
#                 active_loss = attention_mask.view(-1) == 1
#                 active_logits = logits.view(-1, self.num_labels)
#                 active_labels = torch.where(
#                     active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
#                 )
#                 loss = loss_fct(active_logits, active_labels)
                
#             else:
#                 loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
#             outputs = (loss,) + outputs

#         return outputs  # (loss), scores, (hidden_states), (attentions)

        if labels is not None:
            labels_crf = labels.clone()      # copy of labels used for crf loss
            labels_crf[labels_crf==-100] = 0 # remove -100 otherwise not working!
            attention_mask = attention_mask.byte()
            
            loss = -self.crf(logits, labels_crf, mask=attention_mask) # CRF loss is positive
            
            # NOTE: self.crf.decode(logits, mask=attention_mask) is a list of lists
            # In order to use the trainer.predict() fct, convert it into a PADDED TENSOR! 
            max_size = labels_crf.shape[1]
            predictions_ll = self.crf.decode(logits, mask=attention_mask)            # list of lists
            predictions_ll = [ l + [-100]*(max_size-len(l)) for l in predictions_ll] # padded list of lists
            predictions_t = torch.Tensor(predictions_ll).int()                       # tensor
    
            outputs = (loss, predictions_t)

        return outputs  # (loss), scores, (hidden_states), (attentions)
