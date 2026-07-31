import torch
from torch import nn
from torchvision import models
from torchvision.models import ResNet18_Weights


class ResNet18Encoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        hidden_dim: int,
        dropout: float = 0.1,
        use_pretrained: bool = True,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        weights = ResNet18_Weights.IMAGENET1K_V1 if use_pretrained else None
        backbone = models.resnet18(weights=weights)
        if in_channels != 3:
            backbone.conv1 = nn.Conv2d(
                in_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.proj(self.backbone(x))


class ImageEncoder(nn.Module):
    def __init__(self, in_channels: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StreetViewAttentionPooling(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, views, hidden_dim]
        attn_logits = self.score(x).squeeze(-1)
        attn_weights = torch.softmax(attn_logits, dim=1)
        pooled = torch.sum(x * attn_weights.unsqueeze(-1), dim=1)
        return pooled


class MLPEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiModalEncoders(nn.Module):
    def __init__(
        self,
        satellite_in_dim: int,
        street_view_in_dim: int,
        tabular_in_dim: int,
        hidden_dim: int,
        image_encoder_type: str = "resnet18",
        street_view_pooling: str = "attention",
        use_pretrained_image_encoder: bool = True,
        freeze_image_backbone: bool = True,
        use_satellite: bool = True,
        use_street_view: bool = True,
        use_tabular: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.use_satellite = use_satellite
        self.use_street_view = use_street_view
        self.use_tabular = use_tabular
        raw_image_in_channels = 3
        if image_encoder_type == "resnet18":
            image_encoder_cls = ResNet18Encoder
        elif image_encoder_type == "simple_cnn":
            image_encoder_cls = ImageEncoder
        else:
            raise ValueError(f"Unsupported image_encoder_type: {image_encoder_type}")

        if self.use_satellite:
            if image_encoder_type == "resnet18":
                self.satellite = image_encoder_cls(
                    raw_image_in_channels,
                    hidden_dim,
                    dropout,
                    use_pretrained=use_pretrained_image_encoder,
                    freeze_backbone=freeze_image_backbone,
                )
            else:
                self.satellite = image_encoder_cls(raw_image_in_channels, hidden_dim, dropout)
            self.satellite_cached_proj = MLPEncoder(satellite_in_dim, hidden_dim, dropout)
        else:
            self.satellite = None
            self.satellite_cached_proj = None

        if self.use_street_view:
            if image_encoder_type == "resnet18":
                self.street_view = image_encoder_cls(
                    raw_image_in_channels,
                    hidden_dim,
                    dropout,
                    use_pretrained=use_pretrained_image_encoder,
                    freeze_backbone=freeze_image_backbone,
                )
            else:
                self.street_view = image_encoder_cls(raw_image_in_channels, hidden_dim, dropout)
        else:
            self.street_view = None

        self.tabular = MLPEncoder(tabular_in_dim, hidden_dim, dropout) if self.use_tabular else None
        self.street_view_pooling = street_view_pooling
        self.street_view_attention = StreetViewAttentionPooling(hidden_dim, dropout) if self.use_street_view else None

    def forward(
        self,
        satellite: torch.Tensor,
        street_view: torch.Tensor,
        tabular: torch.Tensor,
        satellite_embedding: torch.Tensor = None,
    ):
        encoded = {}

        if self.use_satellite:
            if satellite_embedding is not None and satellite_embedding.numel() > 0:
                encoded["satellite"] = self.satellite_cached_proj(satellite_embedding)
            else:
                encoded["satellite"] = self.satellite(satellite)

        if self.use_street_view:
            batch_size, num_views, channels, height, width = street_view.shape
            street_view_flat = street_view.view(batch_size * num_views, channels, height, width)
            street_view_encoded = self.street_view(street_view_flat)
            street_view_encoded = street_view_encoded.view(batch_size, num_views, -1)

            if self.street_view_pooling == "attention":
                street_view_encoded = self.street_view_attention(street_view_encoded)
            elif self.street_view_pooling == "mean":
                street_view_encoded = street_view_encoded.mean(dim=1)
            else:
                raise ValueError(f"Unsupported street_view_pooling: {self.street_view_pooling}")
            encoded["street_view"] = street_view_encoded

        if self.use_tabular:
            encoded["tabular"] = self.tabular(tabular)

        return encoded
