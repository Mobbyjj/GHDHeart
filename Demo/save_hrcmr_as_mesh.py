# load all the files under the HRUKBB dire
import os
import sys 
sys.path.append(os.path.join('..'))


import numpy as np
import torch
import torch.nn.functional as F
import data.data_utils as dut
from data_process.dataset_real_scaling import *
from GHD.GHD_cardiac import GHD_Cardiac
from GHD import GHD_config
from ops.medical_related import get_4chamberview_frame
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.io import save_obj

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

datadir = '/media/ssd/fanwen/MultiView/HRUKBB/Dataset'
rootdir = '/media/ssd/fanwen/MultiView/HRUKBB/'
savedir = '/media/ssd/fanwen/MultiView/HRUKBB/GHBMesh'

HRcases = ['HR_ES', 'HR_ED']
center_aligned = True
rescalar = 1/100
B = 1
sample_num = 2000


casenames = os.listdir(datadir)

root_path = os.path.dirname(os.path.realpath('.'))
base_shape_path = 'canonical_shapes/Standard_LV_2000.obj'
base_shape_path = os.path.join(root_path, base_shape_path)
bi_ventricle_path = 'canonical_shapes/Standard_BiV.obj'
bi_ventricle_path = os.path.join(root_path, bi_ventricle_path)

cfg = GHD_config(base_shape_path=base_shape_path,
            num_basis=6**2, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
            device='cuda:3',
            if_nomalize=True, if_return_scipy=True, 
            bi_ventricle_path=bi_ventricle_path)

paraheart = GHD_Cardiac(cfg) #

for casename in casenames:
    casedir = os.path.join(datadir, casename)
    # sample the HR_ED segs
    for HRcase in HRcases:
        try:
            if not os.path.exists(os.path.join(casedir, (HRcase + '.nii.gz'))):
                continue
            hr_nii_path = os.path.join(casedir, (HRcase + '.nii.gz'))
            output = dut.load_nib_image(hr_nii_path)
            label_torch = torch.from_numpy(output['img']).permute(2,1,0).unsqueeze(0).float()
            affine_torch = torch.from_numpy(output['affine']).float()
            window_size = torch.from_numpy(output['window_size']).float()

            affine_torch2np = torch.diag(torch.tensor([label_torch.shape[-1]-1.,
                                          label_torch.shape[-2]-1.,
                                           label_torch.shape[-3]-1.,
                                           1.])/2)
            if center_aligned:
                affine_torch[:3, 3] = 0 # set the translation to 0
            else:
                affine_torch2np[:3, 3] = torch.tensor([label_torch.shape[-1]-1.,
                                                    label_torch.shape[-2]-1.,
                                                    label_torch.shape[-3]-1.])/2.
                
            affine_torch = affine_torch@affine_torch2np
            affine_torch[:3,:] = affine_torch[:3,:]*rescalar
            affine_torch[3,:] = torch.tensor([0,0,0,1])

            label_tem = label_torch.clone().unsqueeze(1).to(device)

            affine_tem = affine_torch.clone().to(device)
            coordinate_map_tem = dut.get_coord_map_3d_normalized(label_tem.shape[-3:], 
                                                                affine_tem)

            B, C, Z, Y, X = label_tem.shape

            Z_rv, Y_rv, X_rv = torch.where(label_tem[0, 0]==4)
            Z_lv, Y_lv, X_lv = torch.where(label_tem[0, 0]==2)
            Z_cav, Y_cav, X_cav = torch.where(label_tem[0, 0]==1)
            Z_bg, Y_bg, X_bg = torch.where(label_tem[0, 0]==0)
            Z_point, Y_point, X_point = torch.where(label_tem[0, 0]==3)

            Pt_rv = coordinate_map_tem[0, Z_rv, Y_rv, X_rv]
            Pt_lv = coordinate_map_tem[0, Z_lv, Y_lv, X_lv]
            Pt_cav = coordinate_map_tem[0, Z_cav, Y_cav, X_cav]
            Pt_bg = coordinate_map_tem[0, Z_bg, Y_bg, X_bg]

            points_bi = torch.cat([Pt_rv, Pt_lv], dim=0)
            points_lv = Pt_lv
            points_outoflv = torch.cat([Pt_rv, Pt_cav, Pt_bg], dim=0)

            geom_dict = get_4chamberview_frame(Pt_cav, Pt_lv, Pt_rv)
            inital_affine = geom_dict['target_affine']


            bbox_lv = torch.stack([Pt_lv.min(dim=0)[0]-0.05, Pt_lv.max(dim=0)[0]+0.05], dim=-1)

            points_outoflv_in_bbox = points_outoflv[(points_outoflv[:,0]>bbox_lv[0,0]) & (points_outoflv[:,0]<bbox_lv[0,1]) & (points_outoflv[:,1]>bbox_lv[1,0]) & (points_outoflv[:,1]<bbox_lv[1,1]) & (points_outoflv[:,2]>bbox_lv[2,0]) & (points_outoflv[:,2]<bbox_lv[2,1])]

            # ------ finish ------
            paraheart.R = matrix_to_axis_angle(inital_affine[...,:3,:3].to(paraheart.device)).view(paraheart.R.shape)
            paraheart.T = inital_affine[...,:3,3].to(paraheart.device).view(paraheart.T.shape)
            


            mesh_gt_bi_sample = points_bi.detach().cpu().numpy()[np.random.choice(points_bi.shape[0], sample_num, replace=False)]
            paraheart.global_registration_biv(mesh_gt_bi_sample)


            sample_lv = points_lv[np.random.choice(points_lv.shape[0], sample_num, replace=False)]
            sample_outoflv = points_outoflv_in_bbox[np.random.choice(points_outoflv_in_bbox.shape[0], sample_num, replace=False)]
            paraheart.global_registration_lv(sample_lv.detach().cpu().numpy())

            # sample_outoflv = points_outoflv_in_bbox[np.random.choice(points_outoflv_in_bbox.shape[0], sample_num*5, replace=False)]
            convergence, Loss_dict_list  = paraheart.morphing2lvtarget(points_lv, 
                                                                       points_outoflv_in_bbox, 
                                                                       loss_dict = {'Loss_occupancy':1, 'Loss_Laplacian':0.001, 'Loss_thickness': 0.001},
                                                                       lr_start=1e-3, 
                                                                       num_iter=2000, 
                                                                       if_reset=True, 
                                                                       if_fit_R=False, 
                                                                       if_fit_s=True, 
                                                                       if_fit_T=True, 
                                                                       record_convergence=True)

            rotation = paraheart.R.detach().cpu()
            translation = paraheart.T.detach().cpu()

            R = axis_angle_to_matrix(rotation).view(3,3)
            T = translation.view(3,1)

            # here is the affine matrix of the paraheart transformation
            affine = torch.eye(4)
            affine[:3,:3] = R
            affine[:3,3] = T.squeeze()
            affine[3, 3] = 1.0

            affine_in = affine.inverse()

            paraheart.R = torch.Tensor([0, 0, 0]).view(1,3).to(paraheart.device)
            paraheart.T = torch.Tensor([0, 0, 0]).view(1,3).to(paraheart.device)

            out_ghd_mesh = paraheart.rendering()

            # sample the new image slices and the gt mesh.
            affine_trans = affine_in.to(paraheart.device)
            affine_torch_new = affine_trans @ affine_tem
            coordinate_map_tem_new = dut.get_coord_map_3d_normalized(label_tem.shape[-3:], 
                                                                    affine_torch_new)


            Pt_lv_new = coordinate_map_tem_new[0, Z_lv, Y_lv, X_lv]
            Pt_rv_new = coordinate_map_tem_new[0, Z_rv, Y_rv, X_rv]
            Pt_cav_new = coordinate_map_tem_new[0, Z_cav, Y_cav, X_cav]
            Pt_bg_new = coordinate_map_tem_new[0, Z_bg, Y_bg, X_bg]


            points_bi_new = torch.cat([Pt_rv_new, Pt_lv_new], dim=0)
            points_lv_new = Pt_lv_new
            points_outoflv_new = torch.cat([Pt_rv_new, Pt_cav_new, Pt_bg_new], dim=0)

            sample_num = 2000
            sample_lv_new = points_lv_new[np.random.choice(points_lv_new.shape[0], 
                                                        sample_num, replace=False)]
            sample_outoflv_new = points_outoflv_new[np.random.choice(points_outoflv_new.shape[0], 
                                                                    sample_num, replace=False)]

            # new mesh groundtruth, inverse of the affine 
            grid_trans_in = (affine_tem.inverse() @ affine.to(paraheart.device)).unsqueeze(0)[:, :3,:]
            print(grid_trans_in.shape)
            new_affine_grid_in = F.affine_grid(grid_trans_in, 
                                            (1, 1, 200, 200, 200), 
                                            align_corners=True)

            label_torch_iso_cano = F.grid_sample(label_tem, 
                                            new_affine_grid_in, 
                                            mode='nearest', 
                                            padding_mode='zeros', 
                                            align_corners=True)
            mesh_gt_lv_cano = cubify((label_torch_iso_cano==2).squeeze(1).float(), 0.5)

            # save the current_mesh
            save_path = os.path.join(savedir, casename)
            os.makedirs(save_path, exist_ok=True)
            # current_mesh.export(os.path.join(save_path, (HRcase + '_GHD.obj')))
            # 
            verts = current_mesh.verts_list()[0]  # Get vertices as a tensor
            faces = current_mesh.faces_list()[0]  # Get faces as a tensor

            # Convert tensors to float (if necessary)
            verts = verts.float()
            faces = faces.int()

            # Define the output path
            output_path = os.path.join(savedir, casename, (HRcase + '_GHD.obj'))
            # Save as OBJ file
            save_obj(output_path, verts, faces)
            # save the target geom_dict
            np.save(os.path.join(save_path, (HRcase + '_geom_dict.npy')), geom_dict)
        except Exception as e:
            print('Error in case: ', casename, HRcase)
            print(e)
            continue