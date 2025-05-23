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
from ops.medical_related import get_4chamberview_frame_rv
from pytorch3d.transforms import axis_angle_to_matrix, matrix_to_axis_angle
from pytorch3d.io import save_obj

def affine_np2torch(affine_np, 
                    img_size_np, 
                    rescalar = 1/100, 
                    center_aligned = True):
    '''
    convert affine matrix from numpy manner to torch manner (normed real world)
    affine_np: [4, 4] original affine matrix read from the medical image file
    img_size: [3] the size of the image, [x, y, z]
    rescalar: [1] the rescalar of the image, default is 1/100mm, [-100mm, 100mm] -> [-1, 1]
    center_aligned: [bool] whether the image is center aligned, default is True
    '''
    if isinstance(affine_np, np.ndarray):
        affine_np2w = torch.from_numpy(affine_np).float()
    else:
        affine_np2w = affine_np.float()

    affine_torch2np = torch.diag(torch.tensor([img_size_np[0]-1., 
                                               img_size_np[1]-1., 
                                               img_size_np[2]-1., 1.]))/2.
    if center_aligned:
        affine_np2w[:3, 3] = 0 # set the translation to 0
    else:
        affine_torch2np[:3, 3] = torch.tensor([img_size_np[0]-1., 
                                               img_size_np[1]-1., 
                                               img_size_np[2]-1.])/2.
    affine_t2w_ =  affine_np2w @ affine_torch2np
    affine_t2w_[:3,:] = affine_t2w_[:3,:]*rescalar
    affine_t2w_[3,:] = torch.tensor([0,0,0,1])
    return affine_t2w_


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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
base_shape_path = '/home/fanwen/GHDHeart/canonical_shapes/Standard_LV_2000.obj'
base_shape_path = os.path.join(root_path, base_shape_path)
bi_ventricle_path = '/home/fanwen/GHDHeart/canonical_shapes/Standard_BiV.obj'
bi_ventricle_path = os.path.join(root_path, bi_ventricle_path)

cfg = GHD_config(base_shape_path=base_shape_path,
            num_basis=6**2, mix_laplacian_tradeoff={'cotlap':1.0, 'dislap':0.1, 'stdlap':0.1},
            device='cuda:0',
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
            rescalar = 1/100

            affine_t2w_ = affine_np2torch(output['affine'],
                                          output['img'].shape[-3:], 
                                          rescalar=rescalar, 
                                          center_aligned=True)
            affine_t2w_ = affine_t2w_.to(device)
            

            label_tem = label_torch.squeeze(0)

            Z, Y, X = label_tem.shape

            coordinate_map = dut.get_coord_map_3d_normalized(label_tem.shape[-3:], 
                                                            affine_t2w_)

            Z_rv, Y_rv, X_rv = torch.where(label_tem==4)
            Z_lv, Y_lv, X_lv = torch.where(label_tem==2)
            Z_cav, Y_cav, X_cav = torch.where(label_tem==1)
            Z_bg, Y_bg, X_bg = torch.where(label_tem==0)


            Pt_rv = coordinate_map[0, Z_rv, Y_rv, X_rv]
            Pt_lv = coordinate_map[0, Z_lv, Y_lv, X_lv]
            Pt_cav = coordinate_map[0, Z_cav, Y_cav, X_cav]
            Pt_bg = coordinate_map[0, Z_bg, Y_bg, X_bg]

            points_bi = torch.cat([Pt_rv, Pt_lv], dim=0)
            points_lv = Pt_lv
            points_outoflv = torch.cat([Pt_rv, Pt_cav, Pt_bg], dim=0)

            geom_dict = get_4chamberview_frame_rv(Pt_cav, Pt_lv, Pt_rv)
            inital_affine = geom_dict['target_affine']

            bbox_lv = torch.stack([Pt_lv.min(dim=0)[0]-0.05, Pt_lv.max(dim=0)[0]+0.05], dim=-1)

            points_outoflv_in_bbox = points_outoflv[(points_outoflv[:,0]>bbox_lv[0,0]) & (points_outoflv[:,0]<bbox_lv[0,1]) & (points_outoflv[:,1]>bbox_lv[1,0]) & (points_outoflv[:,1]<bbox_lv[1,1]) & (points_outoflv[:,2]>bbox_lv[2,0]) & (points_outoflv[:,2]<bbox_lv[2,1])]

            paraheart.R = matrix_to_axis_angle(inital_affine[...,:3,:3].to(paraheart.device)).view(paraheart.R.shape)
            paraheart.T = inital_affine[...,:3,3].to(paraheart.device).view(paraheart.T.shape)
            
            # no need to reg the bi-ventricle, just use the initial affine
            # mesh_gt_bi_sample = points_bi.detach().cpu().numpy()[np.random.choice(points_bi.shape[0], sample_num, replace=False)]
            # paraheart.global_registration_biv(mesh_gt_bi_sample)

            sample_lv = points_lv[np.random.choice(points_lv.shape[0], sample_num, replace=False)]
            sample_outoflv = points_outoflv_in_bbox[np.random.choice(points_outoflv_in_bbox.shape[0], sample_num, replace=False)]
            paraheart.global_registration_lv(sample_lv.detach().cpu().numpy())

            convergence, Loss_dict_list  = paraheart.morphing2lvtarget(points_lv.to(device), 
                                                                       points_outoflv_in_bbox, 
                                                                       loss_dict = {'Loss_occupancy':1, 'Loss_Laplacian':0.001, 'Loss_thickness': 0.001},
                                                                       lr_start=1e-3, 
                                                                       num_iter=2000, 
                                                                       if_reset=True, 
                                                                       if_fit_R=False, 
                                                                       if_fit_s=True, 
                                                                       if_fit_T=True, 
                                                                       record_convergence=True)
            Dice = 1 - paraheart.dice_evaluation(points_lv, points_outoflv_in_bbox)

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

            current_mesh = paraheart.rendering()
            # save the current_mesh
            save_path = os.path.join(savedir, casename)
            os.makedirs(save_path, exist_ok=True)
            # instead of saving the obj, just save the vertices.
            # output_path = os.path.join(savedir, casename, (HRcase + '_GHD.obj'))

            # If you have a batch of meshes, pick one (say the 0th):
            verts = current_mesh.verts_list()[0]    # (V, 3) tensor
            faces = current_mesh.faces_list()[0]    # (F, 3) tensor

            # Make sure they’re on CPU and floats/ints
            verts = verts.detach().cpu()
            faces = faces.detach().cpu()

            # Write
            # save_obj(output_path, verts, faces)
            geom_dict = {
                    'verts': verts.numpy(), # the vertices of the mesh
                    'faces': faces.numpy(), # the faces of the mesh
                    'affine_target': inital_affine.detach().cpu().numpy(), # the target from canonical -> personal torch affine. 
                    'affine_cano': affine.numpy(),
                    'convergence': convergence,
                    'dice': Dice} # canonical shape -> personal torch                }
            np.save(os.path.join(save_path, (HRcase + '.npy')), geom_dict)
        except Exception as e:
            print('Error in case: ', casename, HRcase)
            print(e)
            continue