import torch
from torch import nn
import numpy as np
from src.models.me2e.modules.e2e_t import MME2E_T

from src.models.me2e.modules.transformer_encoder import WrappedTransformerEncoder
from torchvision import transforms
# from facenet_pytorch import MTCNN
from src.models.me2e.modules.vgg_block import VggBasicBlock


class MModel(nn.Module):
    def __init__(self, model_id, param_model, num_features_text, num_features_image, num_features_audio, num_classes):
        super(MModel, self).__init__()
        self.model_name = 'mme2e'
        self.model_id = model_id
        self.num_features_text = num_features_text
        self.num_features_image = num_features_image
        self.num_features_audio = num_features_audio
        self.num_classes = num_classes
        self.mod = 'tav'
        # self.device = device
        nlayers = param_model['trans_nlayers']
        nheads = param_model['trans_nheads']
        trans_dim = param_model['trans_dim']

        text_cls_dim = 768
        if param_model['text_model_size'] == 'large':
            text_cls_dim = 1024
        if param_model['text_model_size'] == 'xlarge':
            text_cls_dim = 2048

        self.T = MME2E_T(size=param_model['text_model_size'])

        # self.mtcnn = MTCNN(image_size=48, margin=2, post_process=False, device=device)
        self.normalize = transforms.Normalize(mean=[159, 111, 102], std=[37, 33, 32])

        self.V = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.A = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=64),
            VggBasicBlock(in_planes=64, out_planes=128),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=128, out_planes=256),
            nn.MaxPool2d(kernel_size=2, stride=2),
            VggBasicBlock(in_planes=256, out_planes=512),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.v_flatten = nn.Sequential(
            nn.Linear(512 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.a_flatten = nn.Sequential(
            nn.Linear(512 * 8 * 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, trans_dim)
        )

        self.v_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)
        self.a_transformer = WrappedTransformerEncoder(dim=trans_dim, num_layers=nlayers, num_heads=nheads)

        self.v_out = nn.Linear(trans_dim, self.num_classes)
        self.t_out = nn.Linear(text_cls_dim, self.num_classes)
        self.a_out = nn.Linear(trans_dim, self.num_classes)

        self.weighted_fusion = nn.Linear(len(self.mod), 1, bias=False)

    def forward(self, x_text, x_image, x_audio):
        all_logits = []

        if 't' in self.mod:
            text_cls = self.T(x_text, get_cls=True)
            all_logits.append(self.t_out(text_cls))

        if 'v' in self.mod:
            # faces = self.mtcnn(imgs)
            # for i, face in enumerate(faces):
            #     if face is None:
            #         center = self.crop_img_center(torch.tensor(imgs[i]).permute(2, 0, 1))
            #         faces[i] = center
            # faces = [self.normalize(face) for face in faces]
            # faces = torch.stack(faces, dim=0).to(device=self.device)

            imgs_lens = self.get_len_samples(x_image, 'image')
            # faces = self.V(faces)
            imgs = self.V(x_image)
            imgs = self.v_flatten(imgs.flatten(start_dim=1))
            imgs = self.v_transformer(imgs, imgs_lens, get_cls=True)

            all_logits.append(self.v_out(imgs))

        if 'a' in self.mod:

            spec_lens = self.get_len_samples(x_audio, 'audio')
            for a_module in self.A:
                specs = a_module(x_audio)

            specs = self.a_flatten(specs.flatten(start_dim=1))
            specs = self.a_transformer(specs, spec_lens, get_cls=True)
            all_logits.append(self.a_out(specs))

        if len(self.mod) == 1:
            return all_logits[0]

        return self.weighted_fusion(torch.stack(all_logits, dim=-1)).squeeze(-1)

    # def crop_img_center(self, img: torch.tensor, target_size=48):
    #     '''
    #     Some images have un-detectable faces,
    #     to make the training goes normally,
    #     for those images, we crop the center part,
    #     which highly likely contains the face or part of the face.
    #
    #     @img - (channel, height, width)
    #     '''
    #     current_size = img.size(1)
    #     off = (current_size - target_size) // 2 # offset
    #     cropped = img[:, off:off + target_size, off - target_size // 2:off + target_size // 2]
    #     return cropped

    def get_len_samples(self, x_feat, mod_name):
        """
        Return a list of the sequence length of each sample from x_feat, based on the last index with non-zero value.
        
        x_feat: array of shape (num_samples, num_features) where padding has already been performed.
        mod_name: name of the modality ('image' or 'audio')
        """
        def get_len_one_sample(x_feat_i, i, mod_name):
            real_len = np.max(np.nonzero(x_feat_i)) +1  # +1 because we want the length
            if real_len == 0:
                print("Note from mme2e.py: {}-th feature from modality {} is a zero vector.".format(i, mod_name))
            return real_len

        num_samples = x_feat.size(0)
        x = x_feat.numpy()
        lens = [get_len_one_sample(x[i,:], i, mod_name) for i in range(num_samples)]
        return lens
