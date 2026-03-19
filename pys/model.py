import torch
import torch.nn as nn
from transformers import EsmModel

class SigSegmenter(nn.Module):
    def __init__(self, 
                # model_name="facebook/esm2_t6_8M_UR50D", 
                model_name="/home/wangjy/modles/esm2_t6_8M_UR50D",
                num_kingdoms=4,      # Eukarya, Archaea, Neg, Pos
                kingdom_dim=32,      # Kingdom Embedding 维度
                num_classes=6):      # NO_SP, SP, LIPO, TAT, TATLIPO, PILIN
        super().__init__()
        
        # 1. Backbone: ESM-2
        self.esm = EsmModel.from_pretrained(model_name)
        self.d_model = self.esm.config.hidden_size
        
        # 2. Kingdom Embedding
        self.kingdom_embedding = nn.Embedding(num_kingdoms, kingdom_dim)
        
        # 3. Fusion Layer (ESM + Kingdom)
        self.fusion = nn.Sequential(
            nn.Linear(self.d_model + kingdom_dim, self.d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.LayerNorm(self.d_model)
        )
        
        # 4. Context Encoder (整合全局信息)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model, 
            nhead=4, 
            dim_feedforward=self.d_model*4,
            batch_first=True
        )
        self.context_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # 5. Boundary Head (P1: N/H, P2: H/C, P3: CS) Batch, 3, lenght
        self.boundary_predictor = nn.Conv1d(self.d_model, 3, kernel_size=1)
        
        # 6. Global Classification Head
        self.type_classifier = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_classes)
        )

    def forward(self, input_ids, attention_mask, kingdom_ids):
        # 1. ESM Extract
        esm_out = self.esm(input_ids=input_ids, attention_mask=attention_mask)
        x = esm_out.last_hidden_state # [B, 70, D]
        
        # 2. Inject Kingdom Info
        k_emb = self.kingdom_embedding(kingdom_ids) # [B, 32]
        k_emb_expanded = k_emb.unsqueeze(1).expand(-1, x.size(1), -1)
        x = torch.cat([x, k_emb_expanded], dim=-1)
        x = self.fusion(x)
        
        # 3. Context Encode (Mask padding)
        key_padding_mask = (attention_mask == 0)
        x = self.context_encoder(x, src_key_padding_mask=key_padding_mask)
        
        # 4. Predict Boundaries
        boundaries_logits = self.boundary_predictor(x.permute(0, 2, 1)) # [B, 3, 70]
        
        # Masking Output: 强制不预测 Padding 区域
        mask_expanded = attention_mask.unsqueeze(1).expand(-1, 3, -1)
        boundaries_logits = boundaries_logits.masked_fill(mask_expanded == 0, -1e9)
        
        # 5. Predict Global Type (Mean Pooling on valid tokens)
        mask_float = attention_mask.unsqueeze(-1).float() 
        sum_embeddings = torch.sum(x * mask_float, dim=1)
        sum_mask = torch.sum(mask_float, dim=1).clamp(min=1e-9)
        x_pool = sum_embeddings / sum_mask
        
        type_logits = self.type_classifier(x_pool)
        
        return boundaries_logits, type_logits