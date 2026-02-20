"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.cuda.amp import autocast as autocast
from torch.nn import functional as F

from lavis.common.registry import registry
from lavis.models.base_model import all_gather_with_grad, concat_all_gather
from lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from sklearn.cluster import KMeans



def l2norm(X, dim=-1):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt()
    X = torch.div(X, norm)
    return X


def l1norm(X, dim):
    """L1-normalize columns of X"""
    norm = torch.abs(X).sum(dim=dim, keepdim=True)
    X = torch.div(X, norm)
    return 

def info_nce(query, target):
    bs = query.size(0)
    targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(query.device)
    temp = nn.Parameter(0.07 * torch.ones([]))
    x = torch.matmul(query,target).squeeze().to(query.device)
    #print('x',x.shape)
    sim_i2t,_ = x.max(-1)
    sim_i2t = sim_i2t / temp
    return F.cross_entropy(sim_i2t, targets)


@registry.register_model("Blip2QformerCir")
class Blip2QformerCir(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        lambda_scr=0.01, 
        lambda_ctx=0.01,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)


        self.vision_proj_n = nn.Linear(self.Qformer.config.hidden_size, self.Qformer.config.hidden_size)
        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.prompt_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, self.Qformer.config.hidden_size)
        )
        self.prompt_tokens.data.normal_(mean=0.0, std=self.Qformer.config.initializer_range)


        self.lambda_scr = lambda_scr
        self.lambda_ctx = lambda_ctx

        # --- SOP Specific Modules  ---
        
        # 1. Spatial Anchoring Projections 
        # W_q and W_k for attention calculation Eq.(2)
        hidden_size = self.Qformer.config.hidden_size
        self.W_q = nn.Linear(hidden_size, hidden_size)
        self.W_k = nn.Linear(hidden_size, hidden_size)

        # 2. Latent Draft Construction MLP [cite: 32]
        # Maps concatenated [f_focus || f_m] (2D) back to D
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )

        # 3. Gating Function phi(H) for GCCP 
        # Controls the injection of modification increment based on entropy
        self.gate_mlp = nn.Sequential(
            nn.Linear(1, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid() 
        )


    def robust_infoNCE(self,query, target):
        #self.temp.data = torch.clamp(self.temp.data, min=1e-2)
        eps=1e-7
        bs = query.size(0)
        x = torch.matmul(query,target).squeeze().to(query.device)
        sim_i2t,_ = x.max(-1)
        i2t=(sim_i2t/ 0.07).softmax(1)
        i2t = torch.clamp(i2t, min=eps, max=1-eps)
        
        labels = torch.arange(query.shape[0]).long().cuda()
        mask = torch.ones_like(i2t).to(float).to(i2t.device)
        mask[torch.arange(bs), labels] = 0.   
        loss = - ((1. - i2t).log() * mask).sum() / bs
        return loss

    def forward(self, samples, device):
        """
        Forward pass implementing SOP workflow:
        Input:
            samples['image']: Reference Image (x_r)
            samples['text_input']: Modification Text (x_m)
            samples['target']: Target Image (x_t)
        """
        image_r = samples["image"]       # Reference Image [cite: 8]
        image_t = samples["target"]      # Target Image [cite: 8]
        text_mod = samples["text_input"] # Modification Text [cite: 8]
        batch_size = image_r.size(0)
        device = image_r.device

        # ==========================================================
        # Step 0: Base Feature Extraction (BLIP-2 Backbone) [cite: 16]
        # ==========================================================
        
        # A. Reference Image Features (F_r)
        with self.maybe_autocast():
            image_embeds_r = self.ln_vision(self.visual_encoder(image_r))
        image_atts_r = torch.ones(image_embeds_r.size()[:-1], dtype=torch.long).to(device)
        
        # Get Reference Visual Tokens F_r [cite: 17]
        # We need the Q-Former output *before* text fusion for F_r
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output_r = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_r,
            encoder_attention_mask=image_atts_r,
            return_dict=True,
        )
        F_r = query_output_r.last_hidden_state #F.normalize(self.vision_proj(, dim=-1) # Shape: (B, 32, D)
        
        # B. Text Features (f_m) [cite: 18]
        text_tokens = self.tokenizer(
            text_mod, padding="max_length", truncation=True,
            max_length=self.max_txt_len, return_tensors="pt"
        ).to(device)
        
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        # Taking [CLS] token as global text embedding
        f_m = text_output.last_hidden_state[:, 0, :] #F.normalize(self.text_proj(), dim=-1)

        # C. Target Image Features (z_t) [cite: 74]
        # Used for Training Objective
        with self.maybe_autocast():
            image_embeds_t = self.ln_vision(self.visual_encoder(image_t))
        image_atts_t = torch.ones(image_embeds_t.size()[:-1], dtype=torch.long).to(device)
        
        query_output_t = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_t,
            encoder_attention_mask=image_atts_t,
            return_dict=True,
        )
        # Project to metric space
        z_t_ = query_output_t.last_hidden_state #F.normalize(self.vision_proj(), dim=-1)
        z_t = z_t_.mean(1)
        # ==========================================================
        # Module 1: Selective Focus Restoration (SFR) [cite: 11]
        # ==========================================================
        
        # 1. Spatial Anchoring & Attention (Eq. 2) 
        # f_m: (B, D) -> (B, 1, D)
        # F_r: (B, L, D)
        q_proj = self.W_q(f_m).unsqueeze(1) 
        k_proj = self.W_k(F_r)              
        
        # Attention scores
        # (B, 1, D) @ (B, D, L) -> (B, 1, L)
        scores = torch.matmul(q_proj, k_proj.transpose(1, 2)) / (F_r.size(-1) ** 0.5)
        attn_a = F.softmax(scores, dim=-1) # a in Eq. 2, Shape (B, 1, L)
        
        # 2. Ambiguity Quantification (Entropy) (Eq. 3) 
        # H = - sum(a * log(a))
        # Add epsilon to prevent log(0)
        eps = 1e-8
        entropy_H = -torch.sum(attn_a * torch.log(attn_a + eps), dim=-1) # (B, 1)
        
        # 3. Latent Draft Construction (Eq. 4, 5) [cite: 27]
        # f_focus = a * F_r
        f_focus = torch.matmul(attn_a, F_r).squeeze(1) # (B, D)
        
        # Concatenate and project back to D
        f_cat = torch.cat([f_focus, f_m], dim=-1) # (B, 2D)
        z_q = self.fusion_mlp(f_cat) # Latent query (B, D) [cite: 33]
        
        # 4. Structure Consistency Regularization (Eq. 7) [cite: 36]
        # Student distribution (from z_q) vs Teacher distribution (from z_t)
        # Normalize features for cosine similarity
        z_q_norm = F.normalize(z_q, dim=-1)
        
        # Batch-wise similarity matrices
        # sim_student = torch.matmul(z_q_norm, z_q_norm.T) / 0.1#self.temp
        # sim_teacher = torch.matmul(z_t.mean(1), z_t.mean(1).T) / 0.1#self.temp # z_t is already normalized
        
        # # Softmax to get probability distributions p_q and p_t
        # p_student = F.log_softmax(sim_student, dim=-1) # Log-prob for KLDiv
        # p_teacher = F.softmax(sim_teacher, dim=-1)     # Target prob
        
        # # KL Divergence weighted by entropy penalty e^H 
        # kl_loss = F.kl_div(p_student, p_teacher, reduction='batchmean')
        penalty = torch.exp(entropy_H).squeeze() # e^H
        kl_loss = self.kl_div(z_q_norm, z_q_norm, z_t, z_t, 0.1)
        loss_scr = (penalty * kl_loss).mean()

        # ==========================================================
        # Module 2: Orthogonal Subspace Projection (OSP) [cite: 42]
        # ==========================================================
        
        # 1. Invariant Context Anchoring (Eq. 8) [cite: 45]
        # (1 - a) corresponds to non-modified regions
        attn_inv = 1.0 - attn_a # (B, 1, L)
        # f_inv = sum((1-a) * F_r)
        f_inv = torch.matmul(attn_inv, F_r).squeeze(1) # (B, D)
        
        # 2. Orthogonal Decoupling (Eq. 9) 
        # Project z_q onto f_inv direction and subtract it
        # v_mod = z_q - proj(z_q, f_inv)
        
        dot_prod = torch.sum(z_q * f_inv, dim=-1, keepdim=True)
        norm_sq = torch.sum(f_inv * f_inv, dim=-1, keepdim=True)
        
        # Projection of z_q onto f_inv
        proj_component = (dot_prod / (norm_sq + eps)) * f_inv
        
        # Modification increment (orthogonal)
        v_mod = z_q - proj_component # [cite: 57]

        # ==========================================================
        # Module 3: Geometric Composition (GCCP) [cite: 58]
        # ==========================================================
        
        # 1. Uncertainty-Aware Composition (Eq. 11) [cite: 61]
        # Gating function phi(H)
        gate_val = self.gate_mlp(entropy_H) # (B, 1)
        
        # Geometric Query Construction: f_inv + phi * v_mod
        # This acts as the "controlled translation"
        geo_query = f_inv + gate_val * v_mod # (B, D)
        
        # The paper states: Q-Former(f_inv + ..., F_r, F_m)
        # We model this by using the geo_query as the input query embedding to the Q-Former
        # and letting it attend to the visual context (F_r). 
        # Note: Standard Q-Former takes (B, N, D). We expand geo_query.
        geo_query_tokens = geo_query.unsqueeze(1).expand(-1, query_tokens.size(1), -1)
        
        # We need to refine this via Q-Former context interaction
        # Passing new query through Q-Former with visual attention
        compose_output = self.Qformer.bert(
            query_embeds=geo_query_tokens,
            encoder_hidden_states=image_embeds_r, # Interaction with F_r
            encoder_attention_mask=image_atts_r,
            return_dict=True,
        )
        
        # Get final feature (take [CLS] or similar representative token)
        # Using vision_proj to align with z_t dimension
        z_final = compose_output.last_hidden_state[:, 0, :] #F.normalize(self.text_proj(), dim=-1) # 

        # 2. Contextual Fidelity Constraint (Eq. 12) 
        # Proj_finv(z_final) should differ minimaly from f_inv
        # We project z_final onto f_inv
        dot_prod_ctx = torch.sum(z_final * f_inv, dim=-1, keepdim=True)
        norm_sq_ctx = torch.sum(f_inv * f_inv, dim=-1, keepdim=True)
        proj_ctx = (dot_prod_ctx / (norm_sq_ctx + eps)) * f_inv
        
        # We need z_final and f_inv to be in compatible spaces (projected)
        # Assuming f_inv needs to be projected if z_final is projected
        # Ideally, loss is computed in the latent space before final normalization or after.
        # Eq 12 implies L2 norm.
        
        # For consistency, let's assume f_inv is the anchor in the latent space (before vision_proj)
        # But z_final is usually projected. Let's project f_inv via vision_proj for this loss calculation
        f_inv_proj = F.normalize(self.vision_proj_n(f_inv), dim=-1)
        
        # Re-calculate projection with projected vectors
        dot_prod_ctx = torch.sum(z_final * f_inv_proj, dim=-1, keepdim=True)
        norm_sq_ctx = torch.sum(f_inv_proj * f_inv_proj, dim=-1, keepdim=True) # Since normalized, this is 1
        
        proj_z_on_inv = dot_prod_ctx * f_inv_proj
        
        loss_ctx = F.mse_loss(proj_z_on_inv, f_inv_proj) # L2 norm squared equivalent
        query_tokens = self.query_tokens.expand(image_embeds_r.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_embeds_r,
            encoder_attention_mask=image_atts_r,
            return_dict=True,
        )
        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )

        # ==========================================================
        # Optimization Objective (Eq. 14) [cite: 73]
        # ==========================================================
        
        # Retrieval Loss (Batch-based Classification)
        # z_final vs z_t
        z_final = F.normalize(self.text_proj(z_final), dim=-1) + fusion_feats
        z_t_final = F.normalize(self.vision_proj(z_t_), dim=-1)

        fusion_feats_ = z_final.unsqueeze(1).unsqueeze(1)
        fusion_fea = fusion_feats.unsqueeze(1).unsqueeze(1)
        target_feats=z_t_final.permute(0, 2, 1)
        loss_retrieval = self.info_nce(fusion_feats_, target_feats) + self.info_nce(fusion_fea, target_feats)
        # Total Loss
        loss_total = loss_retrieval + self.lambda_scr * loss_scr + self.lambda_ctx * loss_ctx

        return {
            "loss_stu_rank": loss_total,
            "loss_retrieval": loss_retrieval,
            "loss_scr": loss_scr,
            "loss_ctx": loss_ctx
        }
    

    def kl_div(self, x1, y1, x2, y2, t):
        x1 = F.normalize(x1, p=2, dim=-1)
        y1 = F.normalize(y1, p=2, dim=-1)
        x2 = F.normalize(x2, p=2, dim=-1)
        y2 = F.normalize(y2, p=2, dim=-1)

        x1_y1 = torch.mm(x1, y1.T) / t
        x2_y2 = torch.mm(x2, y2.T) / t

        log_soft_x1 = F.log_softmax(x1_y1, dim=1)
        soft_x2 = F.softmax(torch.autograd.Variable(x2_y2), dim=1)
        kl = F.kl_div(log_soft_x1, soft_x2, reduction='batchmean')

        return kl

    def info_nce(self, query, target):
        bs = query.size(0)
        targets = torch.linspace(0,  bs - 1, bs, dtype=int).to(query.device)
        x = torch.matmul(query,target).squeeze().to(query.device)
        #print('x',x.shape)
        sim_i2t,_ = x.max(-1)
        sim_i2t = sim_i2t / self.temp
        return F.cross_entropy(sim_i2t, targets)


    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=3,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples["image"]
        image_embeds = self.ln_vision(self.visual_encoder(image))

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(image.size(0), 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(image.device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image):
        image_embeds = self.ln_vision(self.visual_encoder(image))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(
            image.device
        )

        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        return query_output.last_hidden_state, image_embeds

    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def inference(self, reference_embeds, target_feats, text, return_attns=False):
        reference_embeds = reference_embeds.cuda()
        target_feats = target_feats.cuda()
        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )

        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
            attention_mask=attention_mask,
            return_dict=True,
        )

        fusion_feats = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 32, :]), dim=-1
        )

        sim_t2q = torch.matmul(
            fusion_feats.unsqueeze(1).unsqueeze(1), target_feats.permute(0, 2, 1)
        ).squeeze()

        sim_i2t, _ = sim_t2q.max(-1)
        sim_i2t = sim_i2t / self.temp

        if return_attns:
            return sim_i2t, fusion_output.cross_attentions[6].mean(1)

        return sim_i2t
    
    @torch.no_grad()
    def extract_retrieval_compose(self, img, mod, return_attns=False):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()

        # return image_embeds
        reference_embeds = image_embeds_frozen

        image_atts = torch.ones(reference_embeds.size()[:-1], dtype=torch.long).to(
            reference_embeds.device
        )
        # query tokens
        query_tokens = self.query_tokens.expand(reference_embeds.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            self.device
        )
        # text tokens
        text_tokens = self.tokenizer(
            mod,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(reference_embeds.device)

        attention_mask = torch.cat([query_atts, text_tokens.attention_mask], dim=1)
        fusion_output = self.Qformer.bert(
            text_tokens.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=reference_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=return_attns
        )

        # text_output = self.Qformer.bert(
        #     text_tokens.input_ids,
        #     query_embeds=fusion_output.last_hidden_state[:, : query_tokens.size(1), :],
        #     attention_mask=attention_mask,
        #     return_dict=True,
        # )

        fusion_feats = F.normalize(
            self.text_proj(fusion_output.last_hidden_state[:, 32, :]), dim=-1
        )

        return fusion_feats.unsqueeze(1).unsqueeze(1)

    @torch.no_grad()
    def extract_retrieval_target(self, img):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(img))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
            output_attentions=True
        )
        image_embeds = query_output.last_hidden_state
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features.permute(0, 2, 1)


    @torch.no_grad()
    def extract_target_features(self, image, mode='mean'):
        with self.maybe_autocast():
            image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
        image_embeds_frozen = image_embeds_frozen.float()
        image_atts = torch.ones(
            image_embeds_frozen.size()[:-1], dtype=torch.long
        ).to(self.device)
        query_tokens = self.query_tokens.expand(
            image_embeds_frozen.shape[0], -1, -1
        )

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_frozen,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        image_embeds = query_output.last_hidden_state

        # return image_embeds
        image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)
        return image_features, image_embeds_frozen


    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_input")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self.ln_vision(self.visual_encoder(image))
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
