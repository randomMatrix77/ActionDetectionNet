import torch
import torch.nn as nn
import torchvision
import math
import yaml
import copy


class ImageEncoder(nn.Module):

    def __init__(self):
        super(ImageEncoder, self).__init__()
        self.resnet = torchvision.models.resnet50(pretrained = False)
        layers = [c for c in self.resnet.children()]
        layers = layers[:-2]
        self.im_enc = nn.Sequential(*layers)

    def forward(self, inp):
        return self.im_enc(inp)


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class Transformer(nn.Module):

    def __init__(self, d_model, n_head, n_hid, n_layers, d_out):
        super(Transformer, self).__init__()

        self.d_model = d_model

        self.pe = PositionalEncoding(d_model)
        tf_enc = nn.TransformerEncoderLayer(d_model, n_head, n_hid)
        self.encoder = nn.TransformerEncoder(tf_enc, n_layers)
        self.linear = nn.Linear(d_model, d_out)

    def forward(self, inp):
        inp = inp * math.sqrt(self.d_model)
        inp = self.pe(inp)
        out = self.encoder(inp)
        out = self.linear(out)

        return out


class BridgeNet(nn.Module):

    def __init__(self, in_channels = 2048, out_channels = 1024):

        super(BridgeNet, self).__init__()
        self.conv = nn.Sequential(*[nn.Conv2d(in_channels, out_channels, kernel_size=3), nn.ReLU()])

    def forward(self, inp):

        out = self.conv(inp)
        return out


class Actron(nn.Module):

    def __init__(self, path):
        super(Actron, self).__init__()

        model_cfg = self.open_cfg(path)
        tf_cfg = model_cfg["transformer"]
        im_cfg = model_cfg["image_config"]
        self.num_frames = model_cfg["temporal_config"]["frames"]
        self.latent_image_size = model_cfg["bridgenet"]["latent_image_size"]
        self.latent_in_channels = model_cfg["bridgenet"]["in_channels"]

        self.transformer = Transformer(tf_cfg["d_model"], tf_cfg["n_head"], tf_cfg["n_hid"],
                                       tf_cfg["n_layers"], tf_cfg["d_out"])

        self.conv = [BridgeNet(im_cfg["in_channels"], im_cfg["out_channels"])] * self.num_frames

        self.im_encoder = ImageEncoder()
        self.swap_weights(self.im_encoder)

    def swap_weights(self, from_model):
        self.im_encoder_static = copy.deepcopy(from_model)

    def open_cfg(self, path):
        return yaml.safe_load(open(path, "r"))

    def forward(self, inp_frames):
        with torch.no_grad():
            initial = [self.im_encoder_static.forward(frame).reshape(1, -1) for frame in
                       inp_frames[:int(self.num_frames / 2)]]
            initial = torch.vstack(initial)

            final = [self.im_encoder_static.forward(frame).reshape(1, -1) for frame in
                     inp_frames[-(int(self.num_frames / 2)):]]
            final = torch.vstack(final)

        middle = self.im_encoder.forward(inp_frames[int(self.num_frames / 2) + 1]).reshape(1, -1)

        # frame_set = initial.extend(middle).extend(final)  # change to vstack
        frame_set = torch.vstack([initial, middle, final])
        frame_set = torch.vstack([frame.reshape(1, self.latent_in_channels, self.latent_image_size, self.latent_image_size) for frame in frame_set])

        encoded = [conv_layer(frame.unsqueeze(dim=0)) for frame, conv_layer in zip(frame_set, self.conv)]
        encoded = torch.vstack(encoded).reshape(self.num_frames, -1)
        encoded = encoded.unsqueeze(dim=1)
        out = self.transformer.forward(encoded)
        return out
