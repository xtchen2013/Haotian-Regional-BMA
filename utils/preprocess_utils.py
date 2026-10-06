import torch
import torch.nn as nn
from utils.conditioning_inputs import get_land_mask
from utils.conditioning_inputs import get_orography


class PreProcessor(nn.Module):
    def __init__(self, params, device):
        super(PreProcessor, self).__init__()
        self.params = params
        self.device = device
        imgx, imgy = params.img_size
        static_features = None
        ###################
        # region boundary #
        # 0 N ~ 60 N      #
        # 70 E ~ 140 E    #
        # (241, 281)      #
        # crop npy file   #
        ###################
        # load land_sea_mark
        if self.params.add_landmask:
            with torch.no_grad():
                # get lsm (241, 281)
                lsm = torch.tensor(get_land_mask(params.landmask_path), dtype=torch.long)
                # one hot encode and move channels to front (2, 241, 281)
                lsm = torch.permute(torch.nn.functional.one_hot(lsm), (2, 0, 1)).to(torch.float32)
                lsm = torch.reshape(lsm, (1, lsm.shape[0], lsm.shape[1], lsm.shape[2]))[:, :, :imgx, :imgy]
                if static_features is None:
                    static_features = lsm
                else:
                    static_features = torch.cat([static_features, lsm], dim=1)

        # load orography
        if self.params.add_orography:
            with torch.no_grad():
                # get lsm (241, 281) and reshape to (1, 1, 241, 281)
                oro = torch.tensor(get_orography(params.orography_path), dtype=torch.float32)
                oro = torch.reshape(oro, (1, 1, oro.shape[0], oro.shape[1]))[:, :, :imgx, :imgy]

                # standardization
                eps = 1.0e-6  # 0.000001
                oro = (oro - torch.mean(oro)) / (torch.std(oro) + eps)

                #
                if static_features is None:
                    static_features = oro
                else:
                    static_features = torch.cat([static_features, oro], dim=1)  # (1, 3, 241, 281)

        # buffer static_features
        self.do_add_static_features = static_features is not None
        if self.do_add_static_features:
            self.register_buffer("static_features", static_features, persistent=False)

    def forward(self, data):
        # load data (inp, tar, inp_zen, tar_zen)
        if self.params.add_zenith:
            inp, tar, izen, tzen = map(lambda x: x.to(self.device, dtype=torch.float), data)
            inp = torch.cat([inp, izen], dim=1)            # Concatenate input with zenith angle
        else:
            inp, tar = map(lambda x: x.to(self.device, dtype=torch.float), data)

        # flatten out time dim in target along channel dim
        # b, t, c, h, w = tar.shape
        # tar = tar.view(b, -1, h, w)

        # add land_sea_mark and orography
        if self.do_add_static_features:
            # print(inp.shape)
            # print(self.static_features.shape)
            # when batch size >= GPUs
            static = self.static_features.expand(inp.shape[0], -1, -1, -1)
            inp = torch.cat([inp, static], dim=1)

        # return data
        if self.params.add_zenith:
            # print(inp.shape)
            # print(tar.shape)
            # print(tzen.shape)
            return inp, tar, tzen  # map(lambda x: x.to(self.static_features.device, dtype=torch.float), [inp, tar, tzen])
        else:
            return inp, tar, None  # list(map(lambda x: x.to(self.device, dtype=torch.float), [inp, tar])) + [None]
