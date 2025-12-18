import argparse

import torch
import timm
import pandas as pd
from PIL import Image
import torch.nn as nn
import timm.data


# --------- Heads (same as training) ---------
class LinearHead(nn.Module):
    def __init__(self, in_dim, num_classes):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        return self.fc(x)


class MLPHead(nn.Module):
    def __init__(self, in_dim, num_classes, hidden_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# --------- Load backbone ---------
def load_backbone(checkpoint_path, device):
    backbone = timm.create_model(
        "vit_base_patch14_reg4_dinov2.lvd142m",
        pretrained=False,
        num_classes=0,
    )

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state = ckpt.get("state_dict", ckpt)
    state = {k: v for k, v in state.items()
             if not k.startswith(("head.", "classifier."))}
    backbone.load_state_dict(state, strict=False)

    backbone.to(device).eval()
    for p in backbone.parameters():
        p.requires_grad = False

    return backbone


# --------- Main ---------
def main(args):
    device = torch.device(args.device)

    # load species ids
    species_df = pd.read_csv(args.species_ids)
    idx2sid = species_df["species_id"].tolist()

    # backbone
    backbone = load_backbone(args.checkpoint, device)

    # transforms
    data_cfg = timm.data.resolve_model_data_config(backbone)
    transform = timm.data.create_transform(**data_cfg, is_training=False)

    # head
    feat_dim = 768
    num_classes = len(idx2sid)

    if args.head_type == "linear":
        head = LinearHead(feat_dim, num_classes)
    else:
        head = MLPHead(feat_dim, num_classes)

    ckpt = torch.load(args.head_pth, map_location="cpu")
    head.load_state_dict(ckpt["state_dict"], strict=False)
    head.to(device).eval()

    # image
    img = Image.open(args.img).convert("RGB")
    img = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = backbone(img)
        logits = head(feat)
        pred = logits.argmax(dim=1).item()

    print("Predicted class index:", pred)
    print("Predicted species_id:", idx2sid[pred])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--img", required=True)
    parser.add_argument("--species-ids", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--head-pth", required=True)
    parser.add_argument("--head-type", choices=["linear", "mlp"], required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    main(args)
