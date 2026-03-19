import torch
import torch.nn as nn
import torch.nn.functional as F

class SigSegmenterLoss(nn.Module):
    def __init__(self, sigma=1.5, w_cls=1.0, w_loc=2.0, w_order=0.5):
        super().__init__()
        self.sigma = sigma
        self.w_cls = w_cls
        self.w_loc = w_loc
        self.w_order = w_order
        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.loc_loss = nn.KLDivLoss(reduction='batchmean')

    def generate_gaussian(self, centers, seq_len=70, device='cpu'):
        # centers: [N]
        grid = torch.arange(seq_len, device=device).unsqueeze(0) # [1, 70]
        centers = centers.unsqueeze(1) # [N, 1]
        heatmap = torch.exp(- (grid - centers)**2 / (2 * self.sigma**2))
        return heatmap / (heatmap.sum(dim=1, keepdim=True) + 1e-6)

    def forward(self, bound_logits, type_logits, bound_targets, type_targets):
        # 1. Classification Loss
        l_cls = self.cls_loss(type_logits, type_targets)
        
        # 2. Localization & Order Loss (仅针对正样本)
        pos_mask = (bound_targets[:, 2] != -1) # CS exists
        l_loc = torch.tensor(0., device=type_logits.device)
        l_order = torch.tensor(0., device=type_logits.device)

        if pos_mask.sum() > 0:
            pred_b = bound_logits[pos_mask] # [N_pos, 3, 70]
            true_b = bound_targets[pos_mask] # [N_pos, 3]
            
            # KL Divergence
            log_probs = F.log_softmax(pred_b, dim=-1)
            for i in range(3):
                target_map = self.generate_gaussian(true_b[:, i], device=type_logits.device)
                l_loc += self.loc_loss(log_probs[:, i, :], target_map)
            l_loc /= 3.0
            
            # Structural Order Constraint (Soft Argmax)
            grid = torch.arange(70, device=type_logits.device, dtype=torch.float)
            probs = torch.softmax(pred_b, dim=-1)
            coords = torch.sum(probs * grid, dim=-1) # [N_pos, 3]
            
            # P1 < P2 < P3 Constraint
            # ReLU(P1 - P2 + margin) -> if P1 > P2, penalize
            l_order += F.relu(coords[:, 0] - coords[:, 1] + 1.0).mean()
            l_order += F.relu(coords[:, 1] - coords[:, 2] + 1.0).mean()

        return self.w_cls*l_cls + self.w_loc*l_loc + self.w_order*l_order, \
               {"cls": l_cls.item(), "loc": l_loc.item(), "ord": l_order.item()}