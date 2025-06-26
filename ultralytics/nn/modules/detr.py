import torch
import torch.nn as nn

class DETRHead(nn.Module):
    def __init__(self, hidden_dim=256, num_queries=100, num_classes=3):
        super().__init__()
        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.transformer_decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=8),
            num_layers=6
        )
        self.class_embed = nn.Linear(hidden_dim, num_classes + 1)  # +1 for "no object"
        self.bbox_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
            nn.Sigmoid()  # box values in [0, 1] normalized coords
        )

    def forward(self, features):  # features: [B, C, H, W]
        B, C, H, W = features.shape
        x = features.flatten(2).permute(2, 0, 1)  # [HW, B, C]
        queries = self.query_embed.weight.unsqueeze(1).repeat(1, B, 1)  # [num_queries, B, C]
        tgt = torch.zeros_like(queries)  # zero-init target
        hs = self.transformer_decoder(tgt, x)  # [num_queries, B, C]
        hs = hs.permute(1, 0, 2)  # [B, num_queries, C]
        logits = self.class_embed(hs)        # [B, num_queries, num_classes+1]
        boxes = self.bbox_embed(hs)          # [B, num_queries, 4]
        return logits, boxes
