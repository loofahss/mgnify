import torch
import numpy as np
from sklearn.metrics import matthews_corrcoef
from tqdm import tqdm

# Constants for printing
ID2KINGDOM = {0: "EUKARYA", 1: "ARCHAEA", 2: "NEGATIVE", 3: "POSITIVE"}
ID2TYPE = {0: "NO_SP", 1: "SP", 2: "LIPO", 3: "TAT", 4: "TATLIPO", 5: "PILIN"}

class SignalPEvaluator:
    def __init__(self, model, dataloader, device):
        self.model = model
        self.dataloader = dataloader
        self.device = device

    def evaluate(self):
        self.model.eval()
        self.model.to(self.device)
        
        res = {'g_true': [], 'g_pred': [], 'cs_true': [], 'cs_pred': [], 'king': []}
        cs_dict = {}
        
        with torch.no_grad():
            for batch in tqdm(self.dataloader, desc="Eval"):
                ids = batch['input_ids'].to(self.device)
                mask = batch['attention_mask'].to(self.device)
                kings = batch['kingdom_ids'].to(self.device)
                
                # Inference
                b_logits, t_logits = self.model(ids, mask, kings)
                
                # Global Predictions
                t_preds = torch.argmax(t_logits, dim=1)
                
                # CS Predictions (P3 channel = 2)
                cs_probs = torch.softmax(b_logits, dim=-1)
                cs_preds = torch.argmax(cs_probs[:, 2, :], dim=1)
                # SignalP Rule: If predicted NO_SP, CS is invalid (-1)
                cs_preds[t_preds == 0] = -1 
                
                res['g_true'].append(batch['labels'].numpy())
                res['g_pred'].append(t_preds.cpu().numpy())
                res['cs_true'].append(batch['boundaries'][:, 2].numpy())
                res['cs_pred'].append(cs_preds.cpu().numpy())
                res['king'].append(batch['kingdom_ids'].numpy())
                cs_dict.update({tuple(i): j for i, j in zip(ids.cpu().tolist(), cs_preds.cpu().tolist())})

        # Graceful handling: no validation samples
        if all(len(v) == 0 for v in res.values()):
            return {}

        # Concat
        for k in res:
            res[k] = np.concatenate(res[k])
        return self._compute_metrics(res), cs_dict

    def _compute_metrics(self, r):
        metrics = {}
        for kid in np.unique(r['king']):
            kname = ID2KINGDOM[kid]
            # Subset Kingdom
            mask = (r['king'] == kid)
            g_t, g_p = r['g_true'][mask], r['g_pred'][mask]
            cs_t, cs_p = r['cs_true'][mask], r['cs_pred'][mask]
            
            for tid in np.unique(g_t):
                if tid == 0: continue # Skip NO_SP
                tname = ID2TYPE[tid]
                
                # MCC2 (One-vs-All) 区分出特定信号肽类型和其他所有类型
                metrics[f"{kname}_{tname}_MCC2"] = self._mcc(g_t, g_p, tid)
                
                # MCC1 (One-vs-NoSP) -> Mask other SP types 特定信号肽类型和非信号肽
                sub_mask = np.isin(g_t, [tid, 0])
                metrics[f"{kname}_{tname}_MCC1"] = self._mcc(g_t[sub_mask], g_p[sub_mask], tid)
                
                # CS Precision/Recall (Class-aware)
                for w in [0, 1, 2, 3]:
                    # Mask logic: True CS valid only if True Type is current. 
                    # Pred CS valid only if Pred Type is current (or NO_SP handled)
                    valid_t_cs = cs_t.copy()
                    valid_t_cs[g_t != tid] = -1
                    
                    valid_p_cs = cs_p.copy()
                    # If model predicts LIPO but we evaluate SP, prediction is invalid
                    valid_p_cs[~np.isin(g_p, [tid, 0])] = -1
                    
                    p, rec = self._prec_rec(valid_t_cs, valid_p_cs, w)
                    metrics[f"{kname}_{tname}_Prec_w{w}"] = p
                    metrics[f"{kname}_{tname}_Rec_w{w}"] = rec
        return metrics

    def _mcc(self, t, p, pos_cls):
        return matthews_corrcoef((t == pos_cls).astype(int), (p == pos_cls).astype(int))

    def _prec_rec(self, t, p, w):
        # Both must be valid (!= -1) to count as hit/miss candidate
        mask_t = (t != -1) # Real Positives
        mask_p = (p != -1) # Predicted Positives
        
        # Intersection for correctness check
        both_valid = mask_t & mask_p
        diff = np.abs(t[both_valid] - p[both_valid])
        hits = (diff <= w).sum()
        
        recall = hits / mask_t.sum() if mask_t.sum() > 0 else 0.0
        precision = hits / mask_p.sum() if mask_p.sum() > 0 else 0.0
        return precision, recall