import torch
import torch.nn as nn
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel, BertLMPredictionHead
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel, RobertaModel, RobertaLMHead
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions
import torch.nn.functional as F
import math
import numpy as np

class MLPLayer(nn.Module):
    """
    Head for getting sentence representations over RoBERTa/BERT's CLS representation.
    """

    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense = nn.Linear(in_dim, out_dim)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)   # non-linear activation
        return x


class Similarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
class ArcSimilarity(nn.Module):
    """
    Dot product or cosine similarity
    """

    def __init__(self, temp, margin=0.05):
        super().__init__()
        self.temp = temp
        self.margin = margin
        self.cos = nn.CosineSimilarity(dim=-1)
        
    def calculate_arccos1(self, cos_sim, labels=None):
        theta = torch.acos(torch.clamp(cos_sim, -1, 1))
        
        if labels is None:
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes)
        
        selected_labels = torch.where(
            torch.gt(theta, math.pi - self.margin),
            torch.zeros_like(one_hot_labels),one_hot_labels)    
        
        
        final_theta = torch.where(selected_labels.bool(),
                                    theta + self.margin,
                                    theta)
        
        return torch.cos(final_theta)
    
    def calculate_arccos2(self, cos_sim, labels=None, slabels=None):
        theta = torch.acos(torch.clamp(cos_sim, -1, 1))
        
        if labels is None:
            labels = torch.arange(cos_sim.size(0)).long().to(cos_sim.device)
        num_classes = labels.max().item() + 1
        one_hot_labels = F.one_hot(labels, num_classes)
        
        selected_labels = torch.where(
            torch.gt(theta, self.margin),
            torch.ones_like(one_hot_labels),one_hot_labels) * torch.abs(one_hot_labels - 1)
        
        if slabels is None:
            final_theta = torch.where(selected_labels.bool(),
                                    theta - self.margin,
                                    theta)
            
        else:
            final_theta = torch.where(selected_labels.bool(),
                                    theta - (1-slabels)*self.margin,
                                    theta)
            
        return torch.cos(final_theta)

    def forward(self, x, y, slabels=None):
        return self.calculate_arccos2(self.cos(x, y), slabels=slabels) / self.temp


class BertForCL(BertPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.bert = BertModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

class RobertaForCL(RobertaPreTrainedModel):
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def __init__(self, config, *model_args, **model_kargs):
        super().__init__(config)
        self.args = model_kargs['model_args']
        self.roberta = RobertaModel(config)
        self.pooler = MLPLayer(config.hidden_size, config.hidden_size)
        self.init_weights()

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
    ):
        # Flatten input for encoding
        input_ids = input_ids.view((-1, input_ids.size(-1)))  # (bs * num_sent, len)
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))  # (bs * num_sent len)
        if token_type_ids is not None:
            token_type_ids = token_type_ids.view((-1, token_type_ids.size(-1)))  # (bs * num_sent, len)

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=True,
        )

        pooler_output = self.pooler(outputs.last_hidden_state[:, 0])

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

class ResNetVisnModel(nn.Module):
    def __init__(self, feature_dim,  proj_dim):
        super().__init__()
        self.mlp = MLPLayer(feature_dim, proj_dim) # visual features -> grounding space

    def forward(self, x):
        x = self.mlp(x)
        x = x / x.norm(2, dim=-1, keepdim=True)
        return x
    
class ClipVisnModel(nn.Module):
    def __init__(self, feature_dim,  proj_dim):
        super().__init__()
        self.vmlp = MLPLayer(feature_dim, proj_dim)  # visual features -> grounding space
        self.tmlp = MLPLayer(feature_dim, proj_dim) # textual features -> grounding space
        self.logit_scale = torch.tensor(np.log(1 / 0.05))
        self.loss_fct = nn.CrossEntropyLoss()

    def logit(self, image_features, text_features):
        device = image_features.device
        
        logit_scale = self.logit_scale.exp()
        logits_per_image = logit_scale * image_features @ text_features.t()
        logits_per_text = logits_per_image.t()
        
        #logits_per_image, logits_per_text = self.logit(images, texts)
        ground_truth = torch.arange(logits_per_image.size(0)).to(device)
        total_loss = (self.loss_fct(logits_per_image,ground_truth) + self.loss_fct(logits_per_text,ground_truth))/2
        
        return total_loss

    def forward(self, visn_feat, text_feat):
        visn_feat = self.vmlp(visn_feat)
        visn_feat = visn_feat / visn_feat.norm(2, dim=-1, keepdim=True)
        
        #text_feat = self.vmlp(text_feat) 2
        text_feat = self.tmlp(text_feat)
        text_feat = text_feat / text_feat.norm(2, dim=-1, keepdim=True)
        
        return visn_feat, text_feat, None#self.logit(visn_feat, text_feat)

class MCSE(nn.Module):
    def __init__(self, lang_model, visn_model, args):
        super().__init__()
        self.args = args
        self.lang_model = lang_model
        self.visn_model = visn_model
        self.grounding = MLPLayer(args.hidden_size, args.proj_dim)
        
        self.sim = ArcSimilarity(temp=self.args.temp, margin=args.margin1)
        self.sim_vl = ArcSimilarity(temp=self.args.temp_vl, margin=args.margin2)
        
        self.loss_fct = nn.CrossEntropyLoss()
        
        self.using_threshhold = args.using_threshhold
        if self.using_threshhold:
            print("USING THRESHOLD")

    def forward(self, batch):
        lang_output = self.lang_model(input_ids=batch['input_ids'],
                                      attention_mask=batch['attention_mask'],
                                      token_type_ids=batch['token_type_ids'] if 'position_ids' in batch.keys() else None,
                                      position_ids=batch['position_ids'] if 'position_ids' in batch.keys() else None)

        batch_size = batch['input_ids'].size(0)
        num_sent = batch['input_ids'].size(1)

        # [bs*2, hidden] -> [bs, 2, hidden]
        lang_pooled_output = lang_output.last_hidden_state[:, 0].view((batch_size, num_sent, -1))
        lang_projection = lang_output.pooler_output.view((batch_size, num_sent, -1))  # [bs, 2,  hidden],  output of additional MLP layer

        return lang_pooled_output, lang_projection

    def compute_loss(self, batch, cal_inter=False):
        l_pool, l_proj = self.forward(batch)

        # Separate representation
        z1, z2 = l_proj[:, 0], l_proj[:, 1]  # (bs, hidden)
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))  # (bs, bs)

        labels = torch.arange(cos_sim.size(0)).long().to(self.args.device)  # [0, 1, bs-1]  (bs)
        loss = self.loss_fct(cos_sim, labels)  # unsup: bs-1 negatives

        if not cal_inter:
            return loss

        else:
            v, t, _ = self.visn_model(batch['img'], batch['clip_text_feat'])  # [bs, proj_dim]
            l2v_proj = self.grounding(l_pool)  # [bs, 2, proj_dim],  output for vision grounding
            l2v_proj = l2v_proj / l2v_proj.norm(2, dim=-1, keepdim=True)

            p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]  # (bs, proj)
            cos_sim_p0_v = self.sim_vl(p1.unsqueeze(1), v.unsqueeze(0), slabels=batch['vv_scores'])  # (bs, bs)
            cos_sim_p1_v = self.sim_vl(p2.unsqueeze(1), v.unsqueeze(0), slabels=batch['vv_scores'])
            
            
            p1, p2 = l2v_proj[:, 0], l2v_proj[:, 1]  # (bs, proj)
            cos_sim_p0_t = self.sim_vl(p1.unsqueeze(1), t.unsqueeze(0), slabels=batch['cc_scores'])  # (bs, bs)
            cos_sim_p1_t = self.sim_vl(p2.unsqueeze(1), t.unsqueeze(0), slabels=batch['cc_scores'])
            
            if self.using_threshhold:
                cos_sim_p0_v = cos_sim_p0_v + batch['cv_slabels']
                cos_sim_p1_v = cos_sim_p1_v + batch['cv_slabels'] 
                cos_sim_p0_t = cos_sim_p0_t + batch['vc_slabels']
                cos_sim_p1_t = cos_sim_p1_t + batch['vc_slabels'] 
            
            inter_loss1 = (self.loss_fct(cos_sim_p0_v, labels) + self.loss_fct(cos_sim_p1_v, labels)) / 2
            inter_loss2 = (self.loss_fct(cos_sim_p0_t, labels) + self.loss_fct(cos_sim_p1_t, labels)) / 2

            inter_loss = inter_loss1 + inter_loss2

            return loss, inter_loss