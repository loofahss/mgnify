import torch
import argparse
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from model import SigSegmenter
from dataset_mgy import StreamingFastaDataset


def collate_fn(batch):

    ids = [x[0] for x in batch]
    seqs = [x[1] for x in batch]

    tokens = tokenizer(
        seqs,
        padding=True,
        truncation=True,
        max_length=70,
        return_tensors="pt"
    )

    # Add kingdom_ids - default to EUKARYA (kingdom_id=0)
    kingdom_ids = torch.full((len(batch),), fill_value=0, dtype=torch.long)  # 0=EUKARYA
    tokens['kingdom_ids'] = kingdom_ids

    return ids, tokens


parser = argparse.ArgumentParser()

parser.add_argument("--fasta", required=True)
parser.add_argument("--model", required=True)
parser.add_argument("--output", required=True)
parser.add_argument("--batch_size", type=int, default=64)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(
    "/home/wangjy/modles/esm2_t6_8M_UR50D"
)

dataset = StreamingFastaDataset(args.fasta)

loader = DataLoader(
    dataset,
    batch_size=args.batch_size,
    collate_fn=collate_fn,
    num_workers=4
)

model = SigSegmenter().to(device)

checkpoint = torch.load(args.model, map_location=device)

if "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.eval()

with open(args.output, "w") as f:

    f.write("protein_id\thas_SP\tSP_type\tSP_start\tSP_end\n")

    with torch.no_grad():

        for ids, tokens in loader:

            tokens = {k: v.to(device) for k, v in tokens.items()}

            boundaries_logits, type_logits = model(**tokens)

            # Get signal peptide type predictions
            type_preds = type_logits.argmax(-1).cpu().numpy()
            
            # Get boundary position predictions (P1: N/H, P3: CS)
            # boundaries_logits shape: [B, 3, 70]
            boundary_positions = boundaries_logits.argmax(dim=-1).cpu().numpy()  # [B, 3]
            
            # Map type_id to SP type name
            sp_types = ["NO_SP", "SP", "LIPO", "TAT", "TATLIPO", "PILIN"]

            for i, type_id in enumerate(type_preds):

                if type_id == 0:  # NO_SP
                    has_sp = 0
                    sp_type = "NO_SP"
                    start = -1
                    end = -1
                else:  # Has SP (types 1-5)
                    has_sp = 1
                    sp_type = sp_types[type_id]
                    start = int(boundary_positions[i, 0])  # P1 position (N/H)
                    end = int(boundary_positions[i, 2])    # P3 position (CS)

                f.write(f"{ids[i]}\t{has_sp}\t{sp_type}\t{start}\t{end}\n")