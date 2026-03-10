import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import MobileNet_V3_Large_Weights


class MobileNetV3Student(nn.Module):
    """
    MobileNetV3-Large based student network for quality distillation.

    Input:
        x: [B, 3, H, W]

    Output dict:
        logits: [B, num_classes]
        probs:  [B, num_classes]
        score:  [B, 1] in [0, 1]
        feat:   [B, embed_dim]
    """

    def __init__(
        self,
        num_classes: int = 4,
        embed_dim: int = 256,
        dropout: float = 0.2,
        pretrained: bool = True,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        if pretrained:
            backbone = models.mobilenet_v3_large(
                weights=MobileNet_V3_Large_Weights.DEFAULT
            )
        else:
            backbone = models.mobilenet_v3_large(weights=None)

        self.features = backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        backbone_out_dim = 960

        self.embedding = nn.Sequential(
            nn.Linear(backbone_out_dim, embed_dim),
            nn.Hardswish(),
            nn.Dropout(p=dropout),
        )

        self.cls_head = nn.Linear(embed_dim, num_classes)
        self.reg_head = nn.Linear(embed_dim, 1)

        if freeze_backbone:
            for p in self.features.parameters():
                p.requires_grad = False

        self._init_weights()

    def _init_weights(self):
        for m in [self.embedding, self.cls_head, self.reg_head]:
            for module in m.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, mean=0.0, std=0.01)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0.0)

    def forward(self, x: torch.Tensor):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        feat = self.embedding(x)

        logits = self.cls_head(feat)
        probs = torch.softmax(logits, dim=1)

        score = torch.sigmoid(self.reg_head(feat))

        return {
            "logits": logits,
            "probs": probs,
            "score": score,
            "feat": feat,
        }


if __name__ == "__main__":
    model = MobileNetV3Student(pretrained=False)
    dummy = torch.randn(2, 3, 224, 224)
    outputs = model(dummy)
    print("logits:", outputs["logits"].shape)
    print("probs :", outputs["probs"].shape)
    print("score :", outputs["score"].shape)
    print("feat  :", outputs["feat"].shape)
