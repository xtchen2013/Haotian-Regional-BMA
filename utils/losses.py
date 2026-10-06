# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Tuple
import numpy as np
import torch
from torch import nn
from utils.grids import GridQuadrature
import torch_harmonics as harmonics
import logging
import gc


# from utils.YParams import YParams
# import os
# yaml_config = "./config/swin.yaml"
# config = "swin_region_71var_4stages_32batch_large"
# params = YParams(os.path.abspath(yaml_config), config)


class LossHandler(nn.Module):
    """
    Wrapper class that will handle computing losses.
    """
    def __init__(self, params):
        super(LossHandler, self).__init__()
        # get global image shape
        self.n_future = params.n_future
        self.img_shape = (720, 1440)    # for data loss crop
        self.crop_shape = (params.img_size[0], params.img_size[1])
        self.crop_offset = (120-8, 280-20)
        self.physics = params.physics
        self.max_epochs = params.max_epochs
        # self.max_epochs = 30
        self.epoch = 0
        loss_type = self.loss_type = params.loss
        loss_type = set(loss_type.split())
        # 'weighted absolute temp-std squared geometric l2'
        # print('loss_type', loss_type)


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # this can be tested in more different ways #
        #    consider physical rule constrains      #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        if 'weighted' in loss_type:
            if params.channel_weights == 'auto':
                channel_weights = torch.ones(params.n_out_channels, dtype=torch.float32)
                for c, chn in enumerate(params.channel_names):
                    # surface channels
                    if chn in ['u10m', 'v10m', 'msl']:
                        # channel_weights[c] = 0.1
                        channel_weights[c] = 1.0
                    # track channels
                    elif chn in ['t2m', 'q2m', 'precip']:
                        channel_weights[c] = 1.0
                    # pressure level channels
                    elif chn[0] in ['u', 'v', 'z', 't', 'q']:
                        pressure_level = float(chn[1:])
                        channel_weights[c] = 0.001 * pressure_level
                    else:
                        channel_weights[c] = 0.01
            else:
                channel_weights = torch.Tensor(params.channel_weights).float()
        else:
            channel_weights = torch.ones(params.n_out_channels, dtype=torch.float32)
        # print(channel_weights)
        # renormalize the weights to one
        # --> (1, 71, 1, 1)
        channel_weights = channel_weights.reshape(1, -1, 1, 1)
        channel_weights = channel_weights / torch.sum(channel_weights)


        # squared
        if 'squared' in loss_type:
            squared = True
        else:
            squared = False


        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # needs re-calculate time_diff_stds and global_stds #
        # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        if 'temp-std' in loss_type:
            eps = 1e-6
            global_stds = torch.from_numpy(np.load(params.global_stds_path)).reshape(1, -1, 1, 1)[:, params.out_channels]
            time_diff_stds = np.sqrt(params.dt) * torch.from_numpy(np.load(params.time_diff_stds_path)).reshape(1, -1, 1, 1)[:, params.out_channels]
            time_var_weights = global_stds / (time_diff_stds+eps)
            # time_var_weights = 1 / (time_diff_stds+eps)
            if squared:
                time_var_weights = time_var_weights**2
            channel_weights = channel_weights * time_var_weights
        self.register_buffer('channel_weights', channel_weights)


        # weighting factor for the case of multistep training
        # this is a canonical uniform weight we found to work best
        # depending on the problem, a different weight strategy might work better
        multistep_weight = torch.ones(self.n_future+1, dtype=torch.float32) / float(self.n_future+1)
        multistep_weight = multistep_weight.reshape(-1, 1, 1, 1)
        self.register_buffer('multistep_weight', multistep_weight)


        # absolute position
        if 'absolute' in loss_type:
            absolute = True
        else:
            absolute = False


        # pole mask
        if 'pole-masked' in loss_type:
            pole_mask = 1
        else:
            pole_mask = 0


        # which weights to use
        quadrature_rule_type = "naive"  # default
        if params.model_grid_type == "legendre_gauss":
            quadrature_rule_type = "legendre-gauss"
        # print(quadrature_rule_type)


        # decide which loss to use
        if 'l2' in loss_type:
            if 'geometric' in loss_type:
                self.loss_obj = GeometricLpLoss(self.img_shape,
                                                self.crop_shape,
                                                self.crop_offset,
                                                p=2,
                                                absolute=absolute,
                                                squared=squared,
                                                pole_mask=pole_mask,
                                                quadrature_rule=quadrature_rule_type
                                                )
            else:
                self.loss_obj = GeometricLpLoss(self.img_shape,
                                                self.crop_shape,
                                                self.crop_offset,
                                                p=2,
                                                absolute=absolute,
                                                pole_mask=pole_mask,
                                                jacobian='flat'
                                                )
        # elif 'l1' in loss_type:
        #     if 'geometric' in loss_type:
        #         self.loss_obj = GeometricLpLoss(self.img_shape,
        #                                         self.crop_shape,
        #                                         self.crop_offset,
        #                                         p=1,
        #                                         absolute=absolute,
        #                                         pole_mask=pole_mask,
        #                                         quadrature_rule=quadrature_rule_type
        #                                         )
        #     else:
        #         self.loss_obj = GeometricLpLoss(self.img_shape,
        #                                         self.crop_shape,
        #                                         self.crop_offset,
        #                                         p=1,
        #                                         absolute=absolute,
        #                                         pole_mask=pole_mask,
        #                                         jacobian='flat'
        #                                         )
        # elif 'geometric h1' in loss_type:
        #     self.loss_obj = GeometricH1Loss(self.img_shape, absolute=absolute, squared=squared)
        # else:
        #     raise ValueError(f"Unknown loss function: {self.loss_type}")


    def forward(self, pred: torch.Tensor, tar: torch.Tensor, inp: torch.Tensor):
        chw = self.channel_weights
        # if hasattr(self, "minmax"):
        #     chw = torch.ones_like(self.channel_weights)
        #     chw = chw / torch.sum(chw)
        #     chw += self.channel_weights.abs() / torch.sum(self.channel_weights.abs())
        # else:
        #     chw = self.channel_weights

        # multi-steps
        # chw -> (1, 71, 1, 1)
        # chw * multistep_weight -> (1, 71*n_future, 1, 1)
        if self.training:
            chw = (chw * self.multistep_weight).reshape(1, -1)
        else:
            chw = chw.reshape(1, -1)

        data_loss = self.loss_obj(pred, tar, chw)
        # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # # data loss should be normalized #
        # # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!! #
        # # data_loss = torch.tensor(0.0019)
        if not self.physics:
            return data_loss
        else:
            # !!!!!!!!!!!!!!!!!!!! #
            #    very important    #
            #     re-normlize      #
            # !!!!!!!!!!!!!!!!!!!! #
            # import h5py
            # f = h5py.File("/mnt/data5/hdf5_region/2020.h5")
            # lats = f['latitude'][:]
            # inp = torch.from_numpy(f['fields'][28:28+1, ...]).to(device='cuda:1')
            # # BMA pred
            # f = h5py.File("/mnt/data5/2024010800.h5")
            # pred = torch.from_numpy(f['pred'][0:0+1, ...]).to(device='cuda:1')
            # pred = pred[:, :, crop_offset[0]:crop_offset[0]+crop_shape[0], crop_offset[1]:crop_offset[1]+crop_shape[1]]
            # tar = torch.from_numpy(f['tar'][0:0+1, ...]).to(device='cuda:1')
            # tar = tar[:, :, crop_offset[0]:crop_offset[0]+crop_shape[0], crop_offset[1]:crop_offset[1]+crop_shape[1]]
            means = torch.from_numpy(np.load("./stats_haotian_region/region_71var_means.npy")).to(device=pred.device)
            stds = torch.from_numpy(np.load("./stats_haotian_region/region_71var_stds.npy")).to(device=pred.device)
            # # # inp = (inp - means) / stds
            # # # pred = (pred - means) / stds
            # # # tar = (tar - means) / stds
            # # print(inp.shape)      # (8, 75, 256, 320)
            # # print(tar.shape)      # (8, 71, 256, 320)
            # # print(pred.shape)     # (8, 71, 256, 320)
            # # inp = torch.randn(8, 75, 256, 320).to(device='cuda:1')
            # # pred = torch.randn(8, 71, 256, 320).to(device='cuda:1')
            # # tar = torch.randn(8, 71, 256, 320).to(device='cuda:1')
            # inp = inp[:, :71, :, :] * stds + means
            # pred = pred * stds + means
            # tar = tar * stds + means



            # get u, v, t, q, z
            channel_names = ['u10m', 'v10m', 't2m', 'q2m', 'msl', 'precip',
                             'u50', 'u100', 'u150', 'u200', 'u250', 'u300', 'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000',
                             'v50', 'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600', 'v700', 'v850', 'v925', 'v1000',
                             'z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500', 'z600', 'z700', 'z850', 'z925', 'z1000',
                             't50', 't100', 't150', 't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850', 't925', 't1000',
                             'q50', 'q100', 'q150', 'q200', 'q250', 'q300', 'q400', 'q500', 'q600', 'q700', 'q850', 'q925', 'q1000',
                             ]
            precip_idx = 5
            u_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('u')][1:]
            v_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('v')][1:]
            t_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('t')][1:]
            q_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('q')][1:]
            z_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('z')]



            # multi-steps
            # [:, 0:71, :, :], [:, 71:71*2, :, :], [:, 71*2:71*3, :, :], [:, 71*3:71*4, :, :]
            # print('input shape: ', inp.shape)
            # print('pred shape: ', pred.shape)
            # print('tar shape: ', tar.shape)
            # inp = torch.randn(8, 75*(0+1), 256, 320).to(device='cuda:1')
            # pred = torch.randn(8, 71*(3+1), 256, 320).to(device='cuda:1')
            # tar = torch.randn(8, 71*(3+1), 256, 320).to(device='cuda:1')
            physics_loss_steps = []
            for step in range(1, self.n_future+1+1):
                # torch.cuda.empty_cache()
                # gc.collect()
                # !!!!!!!!!!!!!!!!!!!!!!! #
                # must to be re-normlized #
                # !!!!!!!!!!!!!!!!!!!!!!! #
                # print('Step: ', step)
                if step==1:
                    inp_step = inp[:, 71*(step-1):71*step, :, :]
                    inp_step = inp_step[:, :71, :, :] * stds + means
                    # print('Input Shape: ', inp_step.shape)
                    # print('Input Mean: ', inp_step.mean())
                pred_step = pred[:, 71 *(step-1):71*step, :, :]
                pred_step = pred_step * stds + means
                # print('Pred Shape: ', pred_step.shape)
                # print('Pred Mean: ', pred_step.mean())
                tar_step = tar[:, 71*(step-1):71*step, :, :]
                tar_step = tar_step * stds + means
                # print('Target Shape: ', tar_step.shape)
                # print('Target Mean: ', tar_step.mean())


                precip_pred = pred_step[:, precip_idx, ...]
                # precip_tar = tar_step[:, precip_idx, ...]
                u_pred = pred_step[:, u_idx, ...]
                v_pred = pred_step[:, v_idx, ...]
                t_pred = pred_step[:, t_idx, ...]
                q_pred = pred_step[:, q_idx, ...]
                z_pred = pred_step[:, z_idx, ...]
                u_tar = tar_step[:, u_idx, ...]
                v_tar = tar_step[:, v_idx, ...]
                t_tar = tar_step[:, t_idx, ...]
                q_tar = tar_step[:, q_idx, ...]
                z_tar = tar_step[:, z_idx, ...]
                q_inp = inp_step[:, q_idx, ...]
                # pressure level weight
                pl_weight = torch.from_numpy(np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])*0.001)
                pl_weight = pl_weight.to(device=pred.device)
                pl_weight = pl_weight.view(1, 13, 1, 1)
                # get lats
                crop_offset = (120 - 8, 280 - 20)
                crop_shape = (256, 320)
                lats_region = torch.linspace(90.0, -90.0, 721, device=pred.device)[crop_offset[0]: crop_offset[0] + crop_shape[0]]


                # Precipitation Loss
                # ref: 0.0012
                # precip = - Δq × ρ × Δz
                # ρ = P / (R_d × T)
                # 1.0 ~ 1.3 kg/m³
                pressure_levels = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000],
                                                device=pred.device, dtype=torch.float32)
                pressure = pressure_levels * 100.0
                # t_pred_mean = t_pred.mean(dim=(0,2,3))
                # q_pred_mean = q_pred.mean(dim=(0, 2, 3))
                t_virtual = t_pred * (1 + 0.608 * q_pred)
                rho = pressure.view(1, 13, 1, 1) / (287.058 * t_virtual + 1e-6)
                # Δz = -(R_d × T_v / g) × ln(P_k / P_{k+1})
                ln_pressure_ratio = torch.log(pressure[1:] / pressure[:-1])
                delta_z = (287.058 * t_virtual.mean(dim=(0, 2, 3))[:-1] / 9.80665) * ln_pressure_ratio
                delta_z = torch.cat([delta_z, torch.tensor([1000.0], device=pred.device)])
                # expend to grid
                # rho_expand = rho.view(1, 13, 1, 1).expand_as(q_pred)
                delta_z = delta_z.view(1, 13, 1, 1).expand_as(q_pred)
                # Δq (kg/kg)
                dq = q_pred - q_inp
                precip_from_q = (- dq * rho * delta_z).sum(dim=1)
                # from m to mm
                precip_pred_mm = precip_pred * 1000.0
                precip_res = ((precip_pred_mm - precip_from_q) / 1000.0) ** 2
                non_negative_penalty = torch.relu(-precip_pred_mm / 1000.0) ** 2
                # red warm >= 100, orange warm >= 50
                extreme_penalty = torch.relu((precip_pred_mm - 100) / 1000.0) ** 4
                precip_loss = torch.mean(precip_res * 1 + non_negative_penalty * 1 + extreme_penalty * 0.1)
                while precip_loss>data_loss*0.2:
                    precip_loss*=0.5
                del t_virtual, rho, delta_z, dq, precip_from_q, precip_pred, precip_pred_mm, precip_res, non_negative_penalty, extreme_penalty


                # Divergence Loss #
                # ref: 3.4147e-10
                # ∂u/∂x + ∂v/∂y = 0
                # real ranges: 10⁻⁶ ~ 10⁻⁵ s⁻¹
                # global mean: 10⁻¹¹ ~ 10⁻⁹ s⁻¹
                dx_deg = 0.25     # lon = 111320 m * cos(lat)
                dy_deg = 0.25     # lat = 27795 m
                R_earth = 6371000.0      # earth radius
                phi_rad = torch.deg2rad(lats_region)   # from 62 N ~ -1.75 S
                dx_m = dx_deg * (np.pi / 180) * R_earth * torch.cos(phi_rad)
                dy_m = dy_deg * (np.pi / 180) * R_earth
                dx_m_expand = dx_m.view(1, 1, -1, 1).expand(1, 13, 256, 320)
                # dx_m_expand[0, 0, :, 0] == dx_m_expand[0, 0, :, -1]
                dy_m_expand = dy_m * torch.ones((1, 13, 256, 320), device=pred.device)     # (1, 13, 256, 320)
                # dy_m_expand[0, 0, 0, :] == dy_m_expand[0, 0, -1, :]
                du_dx = torch.gradient(u_pred, dim=-1)[0] / dx_m_expand
                dv_dy = torch.gradient(v_pred, dim=-2)[0] / dy_m_expand
                # for i in range(13):
                #     print('∂u/∂x mean:', du_dx[:, i, ...].mean())
                #     print('∂v/∂y mean:', dv_dy[:, i, ...].mean())
                divergence_res = ((du_dx + dv_dy) ** 2)
                divergence_loss = torch.mean(divergence_res * pl_weight)
                # while divergence_loss<data_loss*0.2:
                #     divergence_loss*=2
                del du_dx, dv_dy, divergence_res


                # !!!!!!!!!!!!!!!!! #
                #    very important #
                #     Energy Loss   #
                # !!!!!!!!!!!!!!!!! #
                # Kinetic Energy (J/kg) (5%)
                # KE = 0.5 * ρ (u² + v² + w²) = 0.5 * (u² + v²)
                # real ranges: 50 ~ 500
                # global mean: 100 ~ 200
                kinetic_energy_pred = 0.5 * (u_pred ** 2 + v_pred ** 2)
                kinetic_energy_tar = 0.5 * (u_tar ** 2 + v_tar ** 2)
                # ke_scale = 200.0
                ke_scale = np.load("/mnt/data2/chenxt/haotian_regional/stats_haotian_region/region_kinetic_energy_scale.npy")
                # ke_scale = np.load("/mnt/langchao_nfs/lidx/haotian_regional/stats_haotian_region/region_kinetic_energy_scale.npy")
                ke_scale = torch.from_numpy(ke_scale).to(device=pred.device).view(1, 13, 1, 1)
                kinetic_energy_pred_scale = kinetic_energy_pred / ke_scale
                kinetic_energy_tar_scale = kinetic_energy_tar / ke_scale
                kinetic_energy_res = (kinetic_energy_pred_scale - kinetic_energy_tar_scale) ** 2
                del kinetic_energy_pred, kinetic_energy_tar, kinetic_energy_pred_scale, kinetic_energy_tar_scale
                # Potential Energy (J/kg) (20%)
                # PE = g * z (geopotential height)
                # real ranges: 4×10⁴ ~ 6×10⁴
                # global mean: 5×10⁴
                # percent: 15% ~ 20%
                potential_energy_pred = 9.80665 * z_pred
                potential_energy_tar = 9.80665 * z_tar
                # pe_scale = 5.5e4
                pe_scale = np.load("/mnt/data2/chenxt/haotian_regional/stats_haotian_region/region_potential_energy_scale.npy")
                # pe_scale = np.load("/mnt/langchao_nfs/lidx/haotian_regional/stats_haotian_region/region_potential_energy_scale.npy")
                pe_scale = torch.from_numpy(pe_scale).to(device=pred.device).view(1, 13, 1, 1)
                potential_energy_pred_scale = potential_energy_pred / pe_scale
                potential_energy_tar_scale = potential_energy_tar / pe_scale
                potential_energy_res = (potential_energy_pred_scale - potential_energy_tar_scale) ** 2
                del potential_energy_pred, potential_energy_tar, potential_energy_pred_scale, potential_energy_tar_scale
                # Internal Energy (J/kg) (80%)
                # IE = c_p * t (K)
                # real ranges: 2.5×10⁵ ~ 3.2×10⁵
                # global mean: 3×10⁵
                # percent: 75% ~ 80%
                internal_energy_pred = 1004.64 * t_pred
                internal_energy_tar = 1004.64 * t_tar
                # ie_scale = 3.0e5
                ie_scale = np.load("/mnt/data2/chenxt/haotian_regional/stats_haotian_region/region_internal_energy_scale.npy")
                # ie_scale = np.load("/mnt/langchao_nfs/lidx/haotian_regional/stats_haotian_region/region_internal_energy_scale.npy")
                ie_scale = torch.from_numpy(ie_scale).to(device=pred.device).view(1, 13, 1, 1)
                internal_energy_pred_scale = internal_energy_pred / ie_scale
                internal_energy_tar_scale = internal_energy_tar / ie_scale
                internal_energy_res = (internal_energy_pred_scale - internal_energy_tar_scale) ** 2
                del internal_energy_pred, internal_energy_tar, internal_energy_pred_scale, internal_energy_tar_scale
                # Latent Energy (J/kg) (10%)
                # LE = l_v * q (kg/kg)
                # real ranges: 0 ~ 5×10⁴
                # global mean: 1×10⁴ ~ 2×10⁴
                # percent: 5% ~ 10%
                latent_energy_pred = (2.501e6 - 2369.5 * (t_pred - 273.15)) * q_pred
                latent_energy_tar = (2.501e6 - 2369.5 * (t_pred - 273.15)) * q_tar
                # le_scale = 2.5e6
                le_scale = np.load("/mnt/data2/chenxt/haotian_regional/stats_haotian_region/region_latent_energy_scale.npy")
                # le_scale = np.load("/mnt/langchao_nfs/lidx/haotian_regional/stats_haotian_region/region_latent_energy_scale.npy")
                le_scale = torch.from_numpy(le_scale).to(device=pred.device).view(1, 13, 1, 1)
                latent_energy_pred_scale = latent_energy_pred / le_scale
                latent_energy_tar_scale = latent_energy_tar / le_scale
                latent_energy_res = (latent_energy_pred_scale - latent_energy_tar_scale) ** 2
                del latent_energy_pred, latent_energy_tar, latent_energy_pred_scale, latent_energy_tar_scale

                # needs more test
                # Energy Loss = Kinetic Energy + Potential Energy + Internal Energy + Latent Energy
                # ref: 0.0011
                energy_res = 0.01*kinetic_energy_res + 0.2*potential_energy_res + internal_energy_res + 0.1*latent_energy_res
                energy_loss = torch.mean(energy_res * pl_weight)
                while energy_loss>data_loss*0.2:
                    energy_loss*=0.5
                del energy_res


                # Mass Loss (ignor w*∂q/∂z)
                # ∂q/∂t + u*∂q/∂x + v*∂q/∂y = 0
                # real ranges: 10⁻¹⁵ ~ 10⁻¹²
                # global mean: 10⁻²⁰ ~ 10⁻¹⁸
                dq_dt = (q_pred - q_inp) / 3600 * 6
                dq_dx = torch.gradient(q_pred, dim=-1)[0] / dx_m_expand
                dq_dy = torch.gradient(q_pred, dim=-2)[0] / dy_m_expand
                advect_x = u_pred * dq_dx
                advect_y = v_pred * dq_dy
                mass_res = (dq_dt + advect_x + advect_y) ** 2
                mass_loss = torch.mean(mass_res * pl_weight)
                # while mass_loss<data_loss*0.2:
                #     mass_loss*=2
                del dq_dt, dq_dx, dq_dy, advect_x, advect_y, mass_res


                # Geostrophic Balance Loss
                # (u - u_geo) + (v - v_geo) = 0
                # u_geo = - (g / f) * ∂Z/∂y
                # v_geo =   (g / f) * ∂Z/∂x
                # real ranges: 5 ~ 20
                # global mean: 5 ~ 40
                omega = 7.292e-5
                coriolis_force = 2 * omega * torch.sin(phi_rad)
                coriolis_force = torch.clamp(coriolis_force, min=1e-5)
                coriolis_force = coriolis_force.view(1, 1, -1, 1).expand(1, 13, 256, 320)
                dz_dx = torch.gradient(z_pred, dim=-1)[0] / dx_m_expand
                dz_dy = torch.gradient(z_pred, dim=-2)[0] / dy_m_expand
                dz_dy = torch.clamp(dz_dy, min=-0.01, max=0.01)
                dz_dx = torch.clamp(dz_dx, min=-0.01, max=0.01)
                u_geo = -(9.80665/coriolis_force) * dz_dy
                v_geo = +(9.80665/coriolis_force) * dz_dx
                u_geo = torch.clamp(u_geo, min=-80.0, max=80.0)
                v_geo = torch.clamp(v_geo, min=-80.0, max=80.0)
                geo_res = ((u_pred - u_geo) ** 2 + (v_pred - v_geo) ** 2)
                wind_res = u_pred ** 2 + v_pred ** 2 + 1e-6
                geo_res = geo_res/wind_res
                geo_res = torch.clamp(geo_res, max=100.0)
                # mask low latitude
                mask = torch.sigmoid(torch.abs(phi_rad) / torch.deg2rad(torch.tensor(10.0)) - 1.0)
                mask = mask.view(1, 1, -1, 1).expand(1, 13, 256, 320).float()
                geo_res = geo_res * (0.1 + 0.9 * mask)
                geo_loss = torch.mean(0.0001*geo_res * pl_weight)
                while geo_loss>data_loss*0.2:
                    geo_loss*=0.5
                del coriolis_force, dz_dx, dz_dy, u_geo, v_geo, wind_res, geo_res


                # dynamic alpha
                # alpha_start = 0.2/5
                alpha_start = 0.0
                # alpha_end   = 1.0/5
                # alpha_end   = 2.0
                alpha_end   = 1.0
                ###########
                # physics #
                ###########
                # if self.max_epochs <= 1:
                #     epoch_ratio = 0.0     # for quick test
                # elif self.epoch >= 33:
                #     epoch_ratio = 0.3
                # else:
                #     epoch_ratio = self.epoch / (100-1)
                #####################
                # physics fine-tune #
                #####################
                # alpha_end = 0.5
                # if self.max_epochs <= 1:
                #     epoch_ratio = 0.0     # for quick test
                # elif self.epoch >= 10-1:
                #     epoch_ratio = 0.3
                # else:
                #     epoch_ratio = self.epoch / (30-1)
                #########################
                # adamW needs more test #
                # smaller alpha_end     #
                # or fixed alpha_end    #
                #########################
                # alpha_end = 0.3
                # if self.max_epochs <= 1:
                #     epoch_ratio = 0.0     # for quick test
                # elif self.epoch >= 100-1:
                #     epoch_ratio = 0.0     # for data convergence
                # else:
                #     epoch_ratio = self.epoch / (self.max_epochs-1)
                ###########################
                # adamW fine-tune 4 steps #
                ###########################
                # alpha_end = 0.3/2         # half for smaller
                # if self.max_epochs <= 1:
                #     epoch_ratio = 0.0     # for quick test
                # elif self.epoch >= 33-1:
                #     epoch_ratio = 0.0     # for data convergence
                # else:
                #     epoch_ratio = self.epoch / (self.max_epochs-1)
                ###########################
                # adamW fine-tune 8 steps #
                ###########################
                alpha_end = 0.05         # half for smaller
                if self.max_epochs <= 1:
                    epoch_ratio = 0.0     # for quick test
                elif self.epoch >= 33-1:
                    epoch_ratio = 0.0     # for data convergence
                else:
                    epoch_ratio = self.epoch / (self.max_epochs-1)
                #########
                # alpha #
                #########
                center = 0.55             # latter for adamW
                steepness = 8.0           # smaller for adamW
                alpha = alpha_start + (alpha_end - alpha_start) * (1.0 / (1.0 + np.exp(-steepness * (epoch_ratio - center))))
                # import matplotlib.pyplot as plt
                # import numpy as np
                # alpha = []
                # for epoch in range(70):
                #     epoch_ratio = epoch / max(1, 70 - 1)
                #     alpha.append(alpha_start + (alpha_end - alpha_start) * (1.0 / (1.0 + np.exp(-steepness * (epoch_ratio - center)))))
                # alpha = np.array(alpha)*4
                # plt.plot(alpha)
                # plt.show()


                # dynamic physics loss
                # start should be (0.4/1.4=28.5%)
                # this will impact start be largger (29%)
                # end should be (3/4=75%)
                # end will be close and smaller (70%)
                # physics_loss_raw = (1.0*precip_loss + 2.0*divergence_loss + 0.5*energy_loss + 0.8*mass_loss + 1.5*geo_loss)
                physics_loss_raw = (precip_loss + divergence_loss + energy_loss + mass_loss + geo_loss)
                while physics_loss_raw > alpha*data_loss:
                    physics_loss_raw *= 0.9
                # print('precip_loss (%): ', str(np.round(1.0*precip_loss.detach().cpu().numpy()/physics_loss_raw.detach().cpu().numpy()*100, 2)))
                # print('divergence_loss (%): ', str(np.round(2.0*divergence_loss.detach().cpu().numpy()/physics_loss_raw.detach().cpu().numpy()*100, 2)))
                # print('energy_loss (%): ', str(np.round(0.5*energy_loss.detach().cpu().numpy()/physics_loss_raw.detach().cpu().numpy()*100, 2)))
                # print('mass_loss (%): ', str(np.round(0.8*mass_loss.detach().cpu().numpy()/physics_loss_raw.detach().cpu().numpy()*100, 2)))
                # print('geo_loss (%): ', str(np.round(1.5*geo_loss.detach().cpu().numpy()/physics_loss_raw.detach().cpu().numpy()*100, 2)))
                # physics_loss = physics_loss_raw + alpha*data_loss.detach()
                physics_loss = physics_loss_raw
                del physics_loss_raw


                # target_loss = alpha * data_loss.detach()
                # lambda_precipitation_dynamic = target_loss / (precip_loss + 1e-20)
                # precip_loss = lambda_precipitation_dynamic * precip_loss
                # # print('precip_loss: ' + str(precip_loss.detach().cpu().numpy()))
                # # lambda_divergence = 0.1
                # lambda_divergence_dynamic = target_loss / (divergence_loss + 1e-20)
                # divergence_loss = lambda_divergence_dynamic * divergence_loss
                # # print('divergence_loss: ' + str(divergence_loss.detach().cpu().numpy()))
                # # lambda_energy = 0.05
                # lambda_energy_dynamic = target_loss / (energy_loss + 1e-20)
                # energy_loss = lambda_energy_dynamic * energy_loss
                # # print('energy_loss: ' + str(energy_loss.detach().cpu().numpy()))
                # # lambda_mass = 0.02
                # lambda_mass_dynamic = target_loss / (mass_loss + 1e-20)
                # mass_loss = lambda_mass_dynamic * mass_loss
                # # print('mass_loss: ' + str(mass_loss.detach().cpu().numpy()))
                # # lambda_geo = 0.05
                # lambda_geo_dynamic = target_loss / (geo_loss + 1e-20)
                # geo_loss = lambda_geo_dynamic * geo_loss
                # # print('geo_loss: ' + str(geo_loss.detach().cpu().numpy()))
                # # Physical Loss 20% ~ 100% for Data Loss
                # physics_loss = precip_loss + divergence_loss + energy_loss + mass_loss + geo_loss
                # print('precip_loss (%): ', str(np.round(precip_loss.detach().cpu().numpy()/data_loss.detach().cpu().numpy() * 100, 2)))
                # print('divergence_loss (%): ', str(np.round(divergence_loss.detach().cpu().numpy()/data_loss.detach().cpu().numpy() * 100, 2)))
                # print('energy_loss (%): ', str(np.round(energy_loss.detach().cpu().numpy()/data_loss.detach().cpu().numpy() * 100, 2)))
                # print('mass_loss (%): ', str(np.round(mass_loss.detach().cpu().numpy()/data_loss.detach().cpu().numpy() * 100, 2)))
                # print('geo_loss (%): ', str(np.round(geo_loss.detach().cpu().numpy()/data_loss.detach().cpu().numpy() * 100, 2)))


                # logging.info("Data Loss: " + str(data_loss.detach().cpu().numpy()))
                # logging.info("Physics Loss: " + str(physics_loss.detach().cpu().numpy()))
                # logging.info("Percent: " + str(np.round(physics_loss.detach().cpu().numpy() / (data_loss.detach().cpu().numpy()) * 100, 2)) + "%")
                physics_loss_steps.append(physics_loss)

            # ##############
            # # set to CPU #
            # ##############
            # # !!!!!!!!!!!!!!!!!!!! #
            # #    very important    #
            # #     re-normlize      #
            # # !!!!!!!!!!!!!!!!!!!! #
            # means = torch.from_numpy(np.load("./stats_haotian_region/region_71var_means.npy")).to(device='cpu')
            # stds = torch.from_numpy(np.load("./stats_haotian_region/region_71var_stds.npy")).to(device='cpu')
            # # get u, v, t, q, z
            # channel_names = ['u10m', 'v10m', 't2m', 'q2m', 'msl', 'precip',
            #                  'u50', 'u100', 'u150', 'u200', 'u250', 'u300', 'u400', 'u500', 'u600', 'u700', 'u850', 'u925', 'u1000',
            #                  'v50', 'v100', 'v150', 'v200', 'v250', 'v300', 'v400', 'v500', 'v600', 'v700', 'v850', 'v925', 'v1000',
            #                  'z50', 'z100', 'z150', 'z200', 'z250', 'z300', 'z400', 'z500', 'z600', 'z700', 'z850', 'z925', 'z1000',
            #                  't50', 't100', 't150', 't200', 't250', 't300', 't400', 't500', 't600', 't700', 't850', 't925', 't1000',
            #                  'q50', 'q100', 'q150', 'q200', 'q250', 'q300', 'q400', 'q500', 'q600', 'q700', 'q850', 'q925', 'q1000',
            #                  ]
            # precip_idx = 5
            # u_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('u')][1:]
            # v_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('v')][1:]
            # t_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('t')][1:]
            # q_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('q')][1:]
            # z_idx = [i for i, chn in enumerate(channel_names) if chn.startswith('z')]
            # # multi-steps
            # physics_loss_steps = []
            # for step in range(1, self.n_future+1+1):
            #     if step==1:
            #         inp_step = inp.to('cpu')[:, 71*(step-1):71*step, :, :]
            #         inp_step = inp_step[:, :71, :, :] * stds + means
            #     pred_step = pred.to('cpu')[:, 71 *(step-1):71*step, :, :]
            #     pred_step = pred_step * stds + means
            #     tar_step = tar.to('cpu')[:, 71*(step-1):71*step, :, :]
            #     tar_step = tar_step * stds + means
            #     # get vars
            #     precip_pred = pred_step[:, precip_idx, ...]
            #     # precip_tar = tar_step[:, precip_idx, ...]
            #     u_pred = pred_step[:, u_idx, ...]
            #     v_pred = pred_step[:, v_idx, ...]
            #     t_pred = pred_step[:, t_idx, ...]
            #     q_pred = pred_step[:, q_idx, ...]
            #     z_pred = pred_step[:, z_idx, ...]
            #     u_tar = tar_step[:, u_idx, ...]
            #     v_tar = tar_step[:, v_idx, ...]
            #     t_tar = tar_step[:, t_idx, ...]
            #     q_tar = tar_step[:, q_idx, ...]
            #     z_tar = tar_step[:, z_idx, ...]
            #     q_inp = inp_step[:, q_idx, ...]
            #     # pressure level weight
            #     pl_weight = torch.from_numpy(np.array([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000])*0.001)
            #     pl_weight = pl_weight.to(device='cpu')
            #     pl_weight = pl_weight.view(1, 13, 1, 1)
            #     # get lats
            #     crop_offset = (120 - 8, 280 - 20)
            #     crop_shape = (256, 320)
            #     lats_region = torch.linspace(90.0, -90.0, 721, device='cpu')[crop_offset[0]: crop_offset[0] + crop_shape[0]]
            #
            #
            #     # Precipitation Loss
            #     # ref: 0.0012
            #     # precip = - Δq × ρ × Δz
            #     # ρ = P / (R_d × T)
            #     # 1.0 ~ 1.3 kg/m³
            #     pressure_levels = torch.tensor([50, 100, 150, 200, 250, 300, 400, 500, 600, 700, 850, 925, 1000], device='cpu', dtype=torch.float32)
            #     pressure = pressure_levels * 100.0
            #     # t_pred_mean = t_pred.mean(dim=(0,2,3))
            #     # q_pred_mean = q_pred.mean(dim=(0, 2, 3))
            #     t_virtual = t_pred * (1 + 0.608 * q_pred)
            #     rho = pressure.view(1, 13, 1, 1) / (287.058 * t_virtual + 1e-6)
            #     # Δz = -(R_d × T_v / g) × ln(P_k / P_{k+1})
            #     ln_pressure_ratio = torch.log(pressure[1:] / pressure[:-1])
            #     delta_z = (287.058 * t_virtual.mean(dim=(0, 2, 3))[:-1] / 9.80665) * ln_pressure_ratio
            #     delta_z = torch.cat([delta_z, torch.tensor([1000.0], device='cpu')])
            #     # expend to grid
            #     # rho_expand = rho.view(1, 13, 1, 1).expand_as(q_pred)
            #     delta_z = delta_z.view(1, 13, 1, 1).expand_as(q_pred)
            #     # Δq (kg/kg)
            #     dq = q_pred - q_inp
            #     precip_from_q = (- dq * rho * delta_z).sum(dim=1)
            #     # from m to mm
            #     precip_pred_mm = precip_pred * 1000.0
            #     precip_res = ((precip_pred_mm - precip_from_q) / 1000.0) ** 2
            #     non_negative_penalty = torch.relu(-precip_pred_mm / 1000.0) ** 2
            #     # red warm >= 100, orange warm >= 50
            #     extreme_penalty = torch.relu((precip_pred_mm - 100) / 1000.0) ** 4
            #     precip_loss = torch.mean(precip_res * 1 + non_negative_penalty * 1 + extreme_penalty * 0.1)
            #     while precip_loss>data_loss*0.2:
            #         precip_loss*=0.5
            #     del t_virtual, rho, delta_z, dq, precip_from_q, precip_pred, precip_pred_mm, precip_res, non_negative_penalty, extreme_penalty
            #     gc.collect()
            #
            #     # Divergence Loss #
            #     # ref: 3.4147e-10
            #     # ∂u/∂x + ∂v/∂y = 0
            #     # real ranges: 10⁻⁶ ~ 10⁻⁵ s⁻¹
            #     # global mean: 10⁻¹¹ ~ 10⁻⁹ s⁻¹
            #     dx_deg = 0.25     # lon = 111320 m * cos(lat)
            #     dy_deg = 0.25     # lat = 27795 m
            #     R_earth = 6371000.0      # earth radius
            #     phi_rad = torch.deg2rad(lats_region)   # from 62 N ~ -1.75 S
            #     dx_m = dx_deg * (np.pi / 180) * R_earth * torch.cos(phi_rad)
            #     dy_m = dy_deg * (np.pi / 180) * R_earth
            #     dx_m_expand = dx_m.view(1, 1, -1, 1).expand(1, 13, 256, 320)
            #     # dx_m_expand[0, 0, :, 0] == dx_m_expand[0, 0, :, -1]
            #     dy_m_expand = dy_m * torch.ones((1, 13, 256, 320), device='cpu')     # (1, 13, 256, 320)
            #     # dy_m_expand[0, 0, 0, :] == dy_m_expand[0, 0, -1, :]
            #     du_dx = torch.gradient(u_pred, dim=-1)[0] / dx_m_expand
            #     dv_dy = torch.gradient(v_pred, dim=-2)[0] / dy_m_expand
            #     # for i in range(13):
            #     #     print('∂u/∂x mean:', du_dx[:, i, ...].mean())
            #     #     print('∂v/∂y mean:', dv_dy[:, i, ...].mean())
            #     divergence_res = ((du_dx + dv_dy) ** 2)
            #     divergence_loss = torch.mean(divergence_res * pl_weight)
            #     while divergence_loss>data_loss*0.2:
            #         divergence_loss*=0.5
            #     del du_dx, dv_dy, divergence_res
            #     gc.collect()
            #
            #     # !!!!!!!!!!!!!!!!! #
            #     #    very important #
            #     #     Energy Loss   #
            #     # !!!!!!!!!!!!!!!!! #
            #     # Kinetic Energy (J/kg) (5%)
            #     # KE = 0.5 * ρ (u² + v² + w²) = 0.5 * (u² + v²)
            #     # real ranges: 50 ~ 500
            #     # global mean: 100 ~ 200
            #     kinetic_energy_pred = 0.5 * (u_pred ** 2 + v_pred ** 2)
            #     kinetic_energy_tar = 0.5 * (u_tar ** 2 + v_tar ** 2)
            #     # ke_scale = 200.0
            #     ke_scale = np.load("/mnt/data2/chenxt/haotian_regional/stats_haotian_region/region_kinetic_energy_scale.npy")
            #     # ke_scale = np.load("/mnt/langchao_nfs/lidx/haotian_regional/stats_haotian_region/region_kinetic_energy_scale.npy")
            #     ke_scale = torch.from_numpy(ke_scale).to(device='cpu').view(1, 13, 1, 1)
            #     kinetic_energy_pred_scale = kinetic_energy_pred / ke_scale
            #     kinetic_energy_tar_scale = kinetic_energy_tar / ke_scale
            #     kinetic_energy_res = (kinetic_energy_pred_scale - kinetic_energy_tar_scale) ** 2
            #     del kinetic_energy_pred, kinetic_energy_tar, kinetic_energy_pred_scale, kinetic_energy_tar_scale
            #     gc.collect()
            #     # Potential Energy (J/kg) (20%)
            #     # PE = g * z (geopotential height)
            #     # real ranges: 4×10⁴ ~ 6×10⁴
            #     # global mean: 5×10⁴
            #     # percent: 15% ~ 20%
            #     potential_energy_pred = 9.80665 * z_pred
            #     potential_energy_tar = 9.80665 * z_tar
            #     # pe_scale = 5.5e4
            #     pe_scale = np.load("/mnt/data2/chenxt/haotian_regional/stats_haotian_region/region_potential_energy_scale.npy")
            #     # pe_scale = np.load("/mnt/langchao_nfs/lidx/haotian_regional/stats_haotian_region/region_potential_energy_scale.npy")
            #     pe_scale = torch.from_numpy(pe_scale).to(device='cpu').view(1, 13, 1, 1)
            #     potential_energy_pred_scale = potential_energy_pred / pe_scale
            #     potential_energy_tar_scale = potential_energy_tar / pe_scale
            #     potential_energy_res = (potential_energy_pred_scale - potential_energy_tar_scale) ** 2
            #     del potential_energy_pred, potential_energy_tar, potential_energy_pred_scale, potential_energy_tar_scale
            #     gc.collect()
            #     # Internal Energy (J/kg) (80%)
            #     # IE = c_p * t (K)
            #     # real ranges: 2.5×10⁵ ~ 3.2×10⁵
            #     # global mean: 3×10⁵
            #     # percent: 75% ~ 80%
            #     internal_energy_pred = 1004.64 * t_pred
            #     internal_energy_tar = 1004.64 * t_tar
            #     # ie_scale = 3.0e5
            #     ie_scale = np.load("/mnt/data2/chenxt/haotian_regional/stats_haotian_region/region_internal_energy_scale.npy")
            #     # ie_scale = np.load("/mnt/langchao_nfs/lidx/haotian_regional/stats_haotian_region/region_internal_energy_scale.npy")
            #     ie_scale = torch.from_numpy(ie_scale).to(device='cpu').view(1, 13, 1, 1)
            #     internal_energy_pred_scale = internal_energy_pred / ie_scale
            #     internal_energy_tar_scale = internal_energy_tar / ie_scale
            #     internal_energy_res = (internal_energy_pred_scale - internal_energy_tar_scale) ** 2
            #     del internal_energy_pred, internal_energy_tar, internal_energy_pred_scale, internal_energy_tar_scale
            #     gc.collect()
            #     # Latent Energy (J/kg) (10%)
            #     # LE = l_v * q (kg/kg)
            #     # real ranges: 0 ~ 5×10⁴
            #     # global mean: 1×10⁴ ~ 2×10⁴
            #     # percent: 5% ~ 10%
            #     latent_energy_pred = (2.501e6 - 2369.5 * (t_pred - 273.15)) * q_pred
            #     latent_energy_tar = (2.501e6 - 2369.5 * (t_pred - 273.15)) * q_tar
            #     # le_scale = 2.5e6
            #     le_scale = np.load("/mnt/data2/chenxt/haotian_regional/stats_haotian_region/region_latent_energy_scale.npy")
            #     # le_scale = np.load("/mnt/langchao_nfs/lidx/haotian_regional/stats_haotian_region/region_latent_energy_scale.npy")
            #     le_scale = torch.from_numpy(le_scale).to(device='cpu').view(1, 13, 1, 1)
            #     latent_energy_pred_scale = latent_energy_pred / le_scale
            #     latent_energy_tar_scale = latent_energy_tar / le_scale
            #     latent_energy_res = (latent_energy_pred_scale - latent_energy_tar_scale) ** 2
            #     del latent_energy_pred, latent_energy_tar, latent_energy_pred_scale, latent_energy_tar_scale
            #     gc.collect()
            #     # needs more test
            #     # Energy Loss = Kinetic Energy + Potential Energy + Internal Energy + Latent Energy
            #     # ref: 0.0011
            #     energy_res = 0.01*kinetic_energy_res + 0.2*potential_energy_res + internal_energy_res + 0.1*latent_energy_res
            #     energy_loss = torch.mean(energy_res * pl_weight)
            #     while energy_loss>data_loss*0.2:
            #         energy_loss*=0.5
            #     del energy_res
            #     gc.collect()
            #
            #     # Mass Loss (ignor w*∂q/∂z)
            #     # ∂q/∂t + u*∂q/∂x + v*∂q/∂y = 0
            #     # real ranges: 10⁻¹⁵ ~ 10⁻¹²
            #     # global mean: 10⁻²⁰ ~ 10⁻¹⁸
            #     dq_dt = (q_pred - q_inp) / 3600 * 6
            #     dq_dx = torch.gradient(q_pred, dim=-1)[0] / dx_m_expand
            #     dq_dy = torch.gradient(q_pred, dim=-2)[0] / dy_m_expand
            #     advect_x = u_pred * dq_dx
            #     advect_y = v_pred * dq_dy
            #     mass_res = (dq_dt + advect_x + advect_y) ** 2
            #     mass_loss = torch.mean(mass_res * pl_weight)
            #     while mass_loss>data_loss*0.2:
            #         mass_loss*=0.5
            #     del dq_dt, dq_dx, dq_dy, advect_x, advect_y, mass_res
            #     gc.collect()
            #
            #     # Geostrophic Balance Loss
            #     # (u - u_geo) + (v - v_geo) = 0
            #     # u_geo = - (g / f) * ∂Z/∂y
            #     # v_geo =   (g / f) * ∂Z/∂x
            #     # real ranges: 5 ~ 20
            #     # global mean: 5 ~ 40
            #     omega = 7.292e-5
            #     coriolis_force = 2 * omega * torch.sin(phi_rad)
            #     coriolis_force = torch.clamp(coriolis_force, min=1e-5)
            #     coriolis_force = coriolis_force.view(1, 1, -1, 1).expand(1, 13, 256, 320)
            #     dz_dx = torch.gradient(z_pred, dim=-1)[0] / dx_m_expand
            #     dz_dy = torch.gradient(z_pred, dim=-2)[0] / dy_m_expand
            #     dz_dy = torch.clamp(dz_dy, min=-0.01, max=0.01)
            #     dz_dx = torch.clamp(dz_dx, min=-0.01, max=0.01)
            #     u_geo = -(9.80665/coriolis_force) * dz_dy
            #     v_geo = +(9.80665/coriolis_force) * dz_dx
            #     u_geo = torch.clamp(u_geo, min=-80.0, max=80.0)
            #     v_geo = torch.clamp(v_geo, min=-80.0, max=80.0)
            #     geo_res = ((u_pred - u_geo) ** 2 + (v_pred - v_geo) ** 2)
            #     wind_res = u_pred ** 2 + v_pred ** 2 + 1e-6
            #     geo_res = geo_res/wind_res
            #     geo_res = torch.clamp(geo_res, max=100.0)
            #     # mask low latitude
            #     mask = torch.sigmoid(torch.abs(phi_rad) / torch.deg2rad(torch.tensor(10.0)) - 1.0)
            #     mask = mask.view(1, 1, -1, 1).expand(1, 13, 256, 320).float()
            #     geo_res = geo_res * (0.1 + 0.9 * mask)
            #     geo_loss = torch.mean(0.0001*geo_res * pl_weight)
            #     while geo_loss>data_loss*0.2:
            #         geo_loss*=0.5
            #     del coriolis_force, dz_dx, dz_dy, u_geo, v_geo, wind_res, geo_res
            #     gc.collect()
            #
            #     # dynamic alpha
            #     alpha_start = 0.0
            #     alpha_end   = 1.0
            #     ###########################
            #     # adamW fine-tune 8 steps #
            #     ###########################
            #     alpha_end = 0.05         # half for smaller
            #     if self.max_epochs <= 1:
            #         epoch_ratio = 0.0     # for quick test
            #     elif self.epoch >= 33-1:
            #         epoch_ratio = 0.0     # for data convergence
            #     else:
            #         epoch_ratio = self.epoch / (self.max_epochs-1)
            #     #########
            #     # alpha #
            #     #########
            #     center = 0.55             # latter for adamW
            #     steepness = 8.0           # smaller for adamW
            #     alpha = alpha_start + (alpha_end - alpha_start) * (1.0 / (1.0 + np.exp(-steepness * (epoch_ratio - center))))
            #     physics_loss_raw = (precip_loss + divergence_loss + energy_loss + mass_loss + geo_loss)
            #     while physics_loss_raw > alpha*data_loss:
            #         physics_loss_raw *= 0.9
            #     physics_loss = physics_loss_raw
            #     del physics_loss_raw
            #     physics_loss_steps.append(physics_loss)
            #     gc.collect()


            # get all steps mean
            physics_loss = torch.tensor(0.0)
            physics_loss = torch.mean(torch.stack(physics_loss_steps))
            return data_loss + physics_loss, (physics_loss / data_loss) * 100



# double check if polar optimization has an effect - we use 5 here by default
class GeometricLpLoss(nn.Module):
    """
    Computes the Lp loss on the sphere.
    """
    def __init__(self,
                 img_shape: Tuple[int, int],
                 crop_shape: Tuple[int, int],
                 crop_offset: Tuple[int, int],
                 p: Optional[float] = 2.,
                 size_average: Optional[bool] = True,
                 reduction: Optional[bool] = True,
                 absolute: Optional[bool] = False,
                 squared: Optional[bool] = False,
                 pole_mask: Optional[int] = 0,
                 jacobian: Optional[str] = 's2',
                 quadrature_rule: Optional[str] = 'naive',
                 ):
        super(GeometricLpLoss, self).__init__()
        self.p = p
        self.img_shape = img_shape
        self.crop_shape = crop_shape
        self.crop_offset = crop_offset
        self.reduction = reduction
        self.size_average = size_average
        self.absolute = absolute
        self.squared = squared
        self.pole_mask = pole_mask
        # get the quadrature
        self.quadrature = GridQuadrature(quadrature_rule,
                                         img_shape=self.img_shape,
                                         crop_shape=self.crop_shape,
                                         crop_offset=self.crop_offset,
                                         normalize=True,
                                         pole_mask=self.pole_mask,
                                         )


    def abs(self, pred: torch.Tensor, tar: torch.Tensor, chw: torch.Tensor):
        # pred = torch.randn(1, 71, 256, 320).to(device='cuda')
        # tar = torch.randn(1, 71, 256, 320).to(device='cuda')
        num_examples = pred.size()[0]
        all_norms = self.quadrature(torch.abs(pred-tar)**self.p)
        all_norms = all_norms.reshape(num_examples, -1)
        # squared
        if not self.squared:
            all_norms = all_norms**(1./self.p)

        # apply channel weighting
        all_norms = chw * all_norms
        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)
        return all_norms


    # def rel(self, pred: torch.Tensor, tar: torch.Tensor, chw: torch.Tensor):
    #     num_examples = pred.size()[0]
    #     diff_norms = self.quadrature(torch.abs(pred-tar)**self.p)
    #     diff_norms = diff_norms.reshape(num_examples, -1)
    #     tar_norms = self.quadrature(torch.abs(tar)**self.p)
    #     tar_norms = tar_norms.reshape(num_examples, -1)
    #
    #     # divide the ratios
    #     frac_norms = (diff_norms / tar_norms)
    #
    #     # squared
    #     if not self.squared:
    #         frac_norms = frac_norms**(1./self.p)
    #
    #     # setup return value
    #     retval = chw * frac_norms
    #     if self.reduction:
    #         if self.size_average:
    #             retval = torch.mean(retval)
    #         else:
    #             retval = torch.sum(retval)
    #     return retval

    def forward(self, pred: torch.Tensor, tar: torch.Tensor, chw: torch.Tensor):
        if self.absolute:
            loss = self.abs(pred, tar, chw)
        else:
            loss = self.rel(pred, tar, chw)
        return loss


# double check if polar optimization has an effect - we use 5 here by default
# class GeometricH1Loss(nn.Module):
#     """
#     Computes the weighted H1 loss on the sphere.
#     Alpha is a parameter which balances the respective seminorms.
#     """
#
#     def __init__(self, img_shape: Tuple[int, int], p: Optional[float] = 2., size_average: Optional[bool] = False,
#                  reduction: Optional[bool] = True, absolute: Optional[bool] = False, squared: Optional[bool] = False,
#                  alpha: Optional[float] = 0.5):
#         super(GeometricH1Loss, self).__init__()
#         self.reduction = reduction
#         self.size_average = size_average
#         self.absolute = absolute
#         self.squared = squared
#         self.alpha = alpha
#         self.sht = harmonics.RealSHT(*img_shape, grid='equiangular').float()
#         h1_weights = torch.arange(self.sht.lmax).float()
#         h1_weights = h1_weights * (h1_weights + 1)
#         self.register_buffer("h1_weights", h1_weights)
#
#     def abs(self, pred: torch.Tensor, tar: torch.Tensor):
#         num_examples = pred.size()[0]
#         coeffs = torch.view_as_real(self.sht(pred - tar))
#         coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
#         norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
#         l2_norm2 = norm2.reshape(num_examples, -1).sum(dim=-1)
#         h1_norm2 = (norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)
#
#         # squared
#         if not self.squared:
#             all_norms = self.alpha*torch.sqrt(l2_norm2) + (1 - self.alpha)*torch.sqrt(h1_norm2)
#         else:
#             all_norms = self.alpha*l2_norm2 + (1 - self.alpha)*h1_norm2
#         if self.reduction:
#             if self.size_average:
#                 return torch.mean(all_norms)
#             else:
#                 return torch.sum(all_norms)
#         return all_norms
#
#     def rel(self, pred: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None):
#         num_examples = pred.size()[0]
#         coeffs = torch.view_as_real(self.sht(pred - tar))
#         coeffs = coeffs[..., 0]**2 + coeffs[..., 1]**2
#         norm2 = coeffs[..., :, 0] + 2 * torch.sum(coeffs[..., :, 1:], dim=-1)
#         l2_norm2 = norm2.reshape(num_examples, -1).sum(dim=-1)
#         h1_norm2 = (norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)
#         tar_coeffs = torch.view_as_real(self.sht(tar))
#         tar_coeffs = tar_coeffs[..., 0]**2 + tar_coeffs[..., 1]**2
#         tar_norm2 = tar_coeffs[..., :, 0] + 2 * torch.sum(tar_coeffs[..., :, 1:], dim=-1)
#         tar_l2_norm2 = tar_norm2.reshape(num_examples, -1).sum(dim=-1)
#         tar_h1_norm2 = (tar_norm2 * self.h1_weights).reshape(num_examples, -1).sum(dim=-1)
#
#         # squared
#         if not self.squared:
#             diff_norms = self.alpha*torch.sqrt(l2_norm2) + (1 - self.alpha)*torch.sqrt(h1_norm2)
#             tar_norms = self.alpha*torch.sqrt(tar_l2_norm2) + (1 - self.alpha)*torch.sqrt(tar_h1_norm2)
#         else:
#             diff_norms = self.alpha*l2_norm2 + (1 - self.alpha)*h1_norm2
#             tar_norms = self.alpha*tar_l2_norm2 + (1 - self.alpha)*tar_h1_norm2
#         # setup return value
#         retval = diff_norms / tar_norms
#         if mask is not None:
#             retval = retval * mask
#         if self.reduction:
#             if self.size_average:
#                 if mask is None:
#                     retval = torch.mean(retval)
#                 else:
#                     retval = torch.sum(retval) / torch.sum(mask)
#             else:
#                 retval = torch.sum(retval)
#         return retval
#
#     def forward(self, pred: torch.Tensor, tar: torch.Tensor, mask: Optional[torch.Tensor] = None):
#         if self.absolute:
#             loss = self.abs(pred, tar)
#         else:
#             loss = self.rel(pred, tar, mask)
#         return loss


