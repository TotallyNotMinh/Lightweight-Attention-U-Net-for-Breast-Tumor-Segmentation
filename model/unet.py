import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

# =========================================================
# ATTENTION GATE
# =========================================================

class AttentionGate(nn.Module):

    def __init__(self, F_g, F_l, F_int):

        super().__init__()

        self.Wg = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.Wx = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, g):

        g1 = self.Wg(g)
        x1 = self.Wx(x)

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi


# =========================================================
# ATTENTION U-NET WITH EFFICIENTNET-B0 ENCODER
# =========================================================

class UNet(nn.Module):

    def __init__(self, n_class=1, dropout_p=0.5):

        super().__init__()

        # =================================================
        # PRETRAINED ENCODER
        # =================================================

        self.encoder = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            features_only=True,
            in_chans=1,
            out_indices=(0, 1, 2, 3, 4)
        )

        # EfficientNet-B0 feature channels
        # [16, 24, 40, 112, 320]

        # =================================================
        # DROPOUT
        # =================================================

        self.drop1 = nn.Dropout2d(p=dropout_p)         # deepest block
        self.drop2 = nn.Dropout2d(p=dropout_p)
        self.drop3 = nn.Dropout2d(p=dropout_p * 0.67)  # ~0.2
        self.drop4 = nn.Dropout2d(p=dropout_p * 0.33)  # ~0.1 — lightest, near output

        # =================================================
        # DECODER
        # =================================================

        # -------------------------
        # Decoder Block 1
        # -------------------------

        self.up1 = nn.ConvTranspose2d(320, 112, kernel_size=2, stride=2)

        self.att1 = AttentionGate(F_g=112, F_l=112, F_int=56)

        self.d11 = nn.Conv2d(224, 112, kernel_size=3, padding=1)
        self.bn_d11 = nn.BatchNorm2d(112)

        self.d12 = nn.Conv2d(112, 112, kernel_size=3, padding=1)
        self.bn_d12 = nn.BatchNorm2d(112)

        # -------------------------
        # Decoder Block 2
        # -------------------------

        self.up2 = nn.ConvTranspose2d(112, 40, kernel_size=2, stride=2)

        self.att2 = AttentionGate(F_g=40, F_l=40, F_int=20)

        self.d21 = nn.Conv2d(80, 40, kernel_size=3, padding=1)
        self.bn_d21 = nn.BatchNorm2d(40)

        self.d22 = nn.Conv2d(40, 40, kernel_size=3, padding=1)
        self.bn_d22 = nn.BatchNorm2d(40)

        # -------------------------
        # Decoder Block 3
        # -------------------------

        self.up3 = nn.ConvTranspose2d(40, 24, kernel_size=2, stride=2)

        self.att3 = AttentionGate(F_g=24, F_l=24, F_int=12)

        self.d31 = nn.Conv2d(48, 24, kernel_size=3, padding=1)
        self.bn_d31 = nn.BatchNorm2d(24)

        self.d32 = nn.Conv2d(24, 24, kernel_size=3, padding=1)
        self.bn_d32 = nn.BatchNorm2d(24)

        # -------------------------
        # Decoder Block 4
        # -------------------------

        self.up4 = nn.ConvTranspose2d(24, 16, kernel_size=2, stride=2)

        self.att4 = AttentionGate(F_g=16, F_l=16, F_int=8)

        self.d41 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
        self.bn_d41 = nn.BatchNorm2d(16)

        self.d42 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn_d42 = nn.BatchNorm2d(16)

        # =================================================
        # FINAL UPSAMPLING
        # =================================================

        self.final_up = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2)

        self.d_final1 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn_final1 = nn.BatchNorm2d(16)
        self.d_final2 = nn.Conv2d(16, 16, kernel_size=3, padding=1)
        self.bn_final2 = nn.BatchNorm2d(16)

        # =================================================
        # OUTPUT
        # =================================================

        self.outconv = nn.Conv2d(16, n_class, kernel_size=1)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):

        # =================================================
        # ENCODER
        # =================================================

        features = self.encoder(x)

        x1 = features[0]   # 16 channels
        x2 = features[1]   # 24 channels
        x3 = features[2]   # 40 channels
        x4 = features[3]   # 112 channels
        x5 = features[4]   # 320 channels

        # =================================================
        # DECODER 1
        # =================================================

        d1 = self.up1(x5)
        d1 = F.interpolate(d1, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4_att = self.att1(x4, d1)
        d1 = torch.cat([d1, x4_att], dim=1)

        d1 = self.relu(self.bn_d11(self.d11(d1)))
        d1 = self.relu(self.bn_d12(self.d12(d1)))
        d1 = self.drop1(d1)

        # =================================================
        # DECODER 2
        # =================================================

        d2 = self.up2(d1)
        d2 = F.interpolate(d2, size=x3.shape[-2:], mode='bilinear', align_corners=False)
        x3_att = self.att2(x3, d2)
        d2 = torch.cat([d2, x3_att], dim=1)

        d2 = self.relu(self.bn_d21(self.d21(d2)))
        d2 = self.relu(self.bn_d22(self.d22(d2)))
        d2 = self.drop2(d2)

        # =================================================
        # DECODER 3
        # =================================================

        d3 = self.up3(d2)
        d3 = F.interpolate(d3, size=x2.shape[-2:], mode='bilinear', align_corners=False)
        x2_att = self.att3(x2, d3)
        d3 = torch.cat([d3, x2_att], dim=1)

        d3 = self.relu(self.bn_d31(self.d31(d3)))
        d3 = self.relu(self.bn_d32(self.d32(d3)))
        d3 = self.drop3(d3)

        # =================================================
        # DECODER 4
        # =================================================

        d4 = self.up4(d3)
        d4 = F.interpolate(d4, size=x1.shape[-2:], mode='bilinear', align_corners=False)
        x1_att = self.att4(x1, d4)
        d4 = torch.cat([d4, x1_att], dim=1)

        d4 = self.relu(self.bn_d41(self.d41(d4)))
        d4 = self.relu(self.bn_d42(self.d42(d4)))
        d4 = self.drop4(d4)

        # =================================================
        # FINAL UPSAMPLING
        # =================================================

        d4 = self.final_up(d4)
        d4 = self.relu(self.bn_final1(self.d_final1(d4)))
        d4 = self.relu(self.bn_final2(self.d_final2(d4)))

        # =================================================
        # OUTPUT
        # =================================================

        out = self.outconv(d4)

        return out