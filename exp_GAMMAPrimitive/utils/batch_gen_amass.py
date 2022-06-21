from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import numpy as np
import random
import glob
import os, sys
from scipy.spatial.transform import Rotation as R
import smplx
from human_body_prior.tools.model_loader import load_vposer
import torch.nn.functional as F
import torchgeometry as tgm

sys.path.append(os.getcwd())
# from models.fittingop import RotConverter
from models.baseops import RotConverter


import pdb


class BatchGeneratorAMASSCanonicalized(object):
    def __init__(self,
                amass_data_path,
                amass_subset_name=None,
                sample_rate=3,
                body_repr='cmu_41', #['smpl_params', 'cmu_41', 'ssm2_67', 'joint_location', 'bone_transform' ]
                read_to_ram=True
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.amass_data_path = amass_data_path
        self.amass_subset_name = amass_subset_name
        self.sample_rate = sample_rate
        self.data_list = []
        self.jts_list = []
        self.body_repr = body_repr
        self.read_to_ram = read_to_ram
        self.max_len = 100 if 'x10' in amass_data_path else 10


    def reset(self):
        self.index_rec = 0
        if self.read_to_ram:
            random.shuffle(self.data_list)
            idx_permute = torch.randperm(self.data_all.shape[0])
            self.data_all = self.data_all[idx_permute]
        else:
            random.shuffle(self.rec_list)

    def reset_with_jts(self):
        '''
        - this script is to train the marker regressor with rollout.
        - the ground truth joint location is used to provide the canonical coordinate.
        '''
        self.index_rec = 0
        if self.read_to_ram:
            random.shuffle(self.data_list)
            idx_permute = torch.randperm(self.data_all.shape[0])
            self.data_all = self.data_all[idx_permute]
            self.jts_all = self.jts_all[idx_permute]
        else:
            random.shuffle(self.rec_list)


    def has_next_rec(self):
        if self.read_to_ram:
            if self.index_rec < len(self.data_list):
                return True
            return False
        else:
            if self.index_rec < len(self.rec_list):
                return True
            return False


    def get_rec_list(self, shuffle_seed=None,
                    to_gpu=False):

        if self.amass_subset_name is not None:
            ## read the sequence in the subsets
            self.rec_list = []
            for subset in self.amass_subset_name:
                self.rec_list += glob.glob(os.path.join(self.amass_data_path,
                                                       subset,
                                                       '*.npz'  ))
        else:
            ## read all amass sequences
            self.rec_list = glob.glob(os.path.join(self.amass_data_path,
                                                    '*/*.npz'))

        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rec_list)
        else:
            random.shuffle(self.rec_list) # shuffle recordings, not frames in a recording.
        if self.read_to_ram:
            print('[INFO] read all data to RAM...')
            for rec in self.rec_list:
                with np.load(rec) as data_:
                    framerate = data_['mocap_framerate']
                    if framerate != 120:
                        continue
                    sample_rate = self.sample_rate
                    pose = data_['poses'][::sample_rate,:66] # 156d = 66d+hand
                    transl = data_['trans'][::sample_rate]

                    if np.isnan(pose).any() or np.isinf(pose).any() or np.isnan(transl).any() or np.isinf(transl).any():
                        continue
                    body_cmu_41 = data_['marker_cmu_41'][::sample_rate]
                    body_ssm2_67 = data_['marker_ssm2_67'][::sample_rate]
                    joints = data_['joints'][::sample_rate].reshape([-1,22,3])
                    transf_rotmat = data_['transf_rotmat']
                    transf_transl = data_['transf_transl']
                
                transl = transl[:self.max_len]
                pose = pose[:self.max_len]
                body_cmu_41 = body_cmu_41[:self.max_len]
                body_ssm2_67 = body_ssm2_67[:self.max_len]
                joints = joints[:self.max_len]
                
                vec_to_target, wpath, marker2tarloc_n = self._get_target_feature(joints, body_ssm2_67, transf_rotmat, transf_transl)

                #['smpl_params', 'cmu_41', 'ssm2_67', 'ssm2_67_marker2tarloc', 'joint_location', 'bone_transform' ]
                if self.body_repr == 'smpl_params':
                    body_feature = np.concatenate([transl, pose],axis=-1)
                elif self.body_repr == 'joints':
                    body_feature = joints.reshape([-1,22*3])
                elif self.body_repr == 'cmu_41':
                    body_feature = body_cmu_41.reshape([-1,67*3])
                elif self.body_repr == 'ssm2_67':
                    body_feature = body_ssm2_67.reshape([-1,67*3])
                elif self.body_repr == 'ssm2_67_marker2tarloc':
                    body_feature = np.concatenate([body_ssm2_67.reshape([-1,67*3]),
                                                   marker2tarloc_n.reshape([-1,67*3])],
                                                axis=-1)
                elif self.body_repr == 'bone_transform':
                    joint_loc = joints
                    joint_rot_aa = pose.reshape([-1, 22, 3])
                    body_feature = np.concatenate([joint_loc, joint_rot_aa], axis=-1)
                else:
                    raise NameError('[ERROR] not valid body representation. Terminate')
                self.data_list.append(body_feature)
                self.jts_list.append(joints)

            self.data_all = np.stack(self.data_list,axis=0) #[b,t,d]
            self.jts_all = np.stack(self.jts_list, axis=0) #[b,t, 22, 3]
            if to_gpu:
                self.data_all = torch.cuda.FloatTensor(self.data_all)
                self.jts_all = torch.cuda.FloatTensor(self.jts_all)


    def next_batch(self, batch_size=64):
        batch_data_ = self.data_all[self.index_rec:self.index_rec+batch_size]
        self.index_rec+=batch_size
        batch_data = torch.cuda.FloatTensor(batch_data_).permute(1,0,2) #[t,b,d]

        return batch_data


    def next_batch_with_jts(self, batch_size=64):
        batch_data_ = self.data_all[self.index_rec:self.index_rec+batch_size].permute(1,0,2)
        batch_jts_ = self.jts_all[self.index_rec:self.index_rec+batch_size].permute(1,0,2,3)
        self.index_rec+=batch_size
        return batch_data_, batch_jts_



    def _get_target_feature(self, joints, body_ssm2_67, rotmat=np.eye(3), transl=np.zeros((1,3))):
        '''normalized walking path'''
        wpath = joints[-1:]-joints
        wpath = wpath[:,0,:2] #(t,2)
        wpath_n = wpath/(1e-8+np.linalg.norm(wpath, axis=-1, keepdims=True))
        '''unnormalized target_marker - starting_marker '''
        vec_to_target = body_ssm2_67[-1:]-body_ssm2_67
        '''normalized target_location - starting_marker'''
        target_loc = joints[-1:, 0:1] # the pelvis (1,1,3)
        target_loc[:,:,-1] = target_loc[:,:,-1] - transl[None,...][:,:,-1]
        vec_to_target_loc = target_loc - body_ssm2_67
        vec_to_target_locn = vec_to_target_loc/np.linalg.norm(vec_to_target_loc, axis=-1, keepdims=True)
        return vec_to_target, wpath_n, vec_to_target_locn



    def next_sequence(self):
        '''
        - this function is only for produce files for visualization or testing in some cases
        - compared to next_batch with batch_size=1, this function also outputs metainfo, like gender, body shape, etc.
        '''
        rec = self.rec_list[self.index_rec]
        with np.load(rec) as data:
            framerate = data['mocap_framerate']
            sample_rate = self.sample_rate
            pose = data['poses'][::sample_rate,:66] # 156d = 66d+hand
            transl = data['trans'][::sample_rate]
            gender = data['gender']
            if np.isnan(pose).any() or np.isinf(pose).any() or np.isnan(transl).any() or np.isinf(transl).any():
                return None
            betas = data['betas'][:10]
            body_cmu_41 = data['marker_cmu_41'][::sample_rate]
            body_ssm2_67 = data['marker_ssm2_67'][::sample_rate]
            framerate = data['mocap_framerate']
            joints = data['joints'][::sample_rate].reshape([-1,22,3])
            transf_rotmat = data['transf_rotmat']
            transf_transl = data['transf_transl']

            ## normalized walking path and unnormalized marker to target
            vec_to_target, wpath, marker2tarloc_n = self._get_target_feature(joints, body_ssm2_67)

        if self.body_repr == 'smpl_params':
            body_feature = np.concatenate([transl, pose],axis=-1)
        elif self.body_repr == 'joints':
            body_feature = joints.reshape([-1,22*3])
        elif self.body_repr == 'cmu_41':
            body_feature = body_cmu_41.reshape([-1,41*3])
        elif self.body_repr == 'ssm2_67':
            body_feature = body_ssm2_67.reshape([-1,67*3])
        elif self.body_repr == 'ssm2_67_marker2tarloc':
                    body_feature = np.concatenate([body_ssm2_67.reshape([-1,67*3]),
                                                   marker2tarloc_n.reshape([-1,67*3])],
                                                axis=-1)
        elif self.body_repr == 'bone_transform':
            joint_loc = joints
            joint_rot_aa = pose.reshape([-1, 22, 3])
            body_feature = np.concatenate([joint_loc, joint_rot_aa], axis=-1)
        else:
            raise NameError('[ERROR] not valid body representation. Terminate')

        ## pack output data
        output = {}
        output['betas'] = betas
        output['gender'] = gender
        output['transl'] = transl
        output['glorot'] = pose[:,:3]
        output['poses'] = pose[:,3:]
        output['body_feature'] = body_feature
        output['transf_rotmat'] = transf_rotmat
        output['transf_transl'] = transf_transl
        output['pelvis_loc'] = joints[:,0,:]
        self.index_rec += 1
        return output




    def next_batch_genderselection(self, batch_size=64, gender='male',
                                    batch_first=True):
        '''
        - this function is to select a batch of data with the same gender
        - it not only outputs body features, but also body parameters, and genders
        - note here the "batch_size" indicates the number of sequences
        '''
        batch_betas = []
        batch_transl = []
        batch_glorot = []
        batch_thetas = []
        batch_body_feature = []
        batch_jts = []
        stack_dim = 0 if batch_first else 1

        bb = 0
        while self.has_next_rec():
            rec = self.rec_list[self.index_rec]
            if bb == batch_size:
                break
            else:
                if str(np.load(rec)['gender']) != gender:
                    self.index_rec += 1
                    continue
            with np.load(rec) as data:
                framerate = data['mocap_framerate']
                sample_rate = int(framerate//self.fps)
                transl = data['trans'][::sample_rate]
                pose = data['poses'][::sample_rate,:66] # 156d = 66d+hand
                betas  = np.tile(data['betas'][:10], (transl.shape[0],1) )
                body_cmu_41 =  data['marker_cmu_41'][::sample_rate]
                body_ssm2_67 = data['marker_ssm2_67'][::sample_rate]
                joints = data['joints'][::sample_rate].reshape([-1,22,3])

            ## normalized walking path and unnormalized marker to target
            vec_to_target, wpath, marker2tarloc_n = self._get_target_feature(joints, body_ssm2_67)

            if self.body_repr == 'smpl_params':
                body_feature = np.concatenate([transl, pose],axis=-1)
            elif self.body_repr == 'joints':
                body_feature = joints.reshape([-1,22*3])
            elif self.body_repr == 'cmu_41':
                body_feature = body_cmu_41.reshape([-1,41*3])
            elif self.body_repr == 'ssm2_67':
                body_feature = body_ssm2_67.reshape([-1,67*3])
            elif self.body_repr == 'ssm2_67_marker2tarloc':
                        body_feature = np.concatenate([body_ssm2_67.reshape([-1,67*3]),
                                                    marker2tarloc_n.reshape([-1,67*3])],
                                                    axis=-1)
            elif self.body_repr == 'bone_transform':
                joint_loc = joints
                joint_rot_aa = pose.reshape([-1, 22, 3])
                body_feature = np.concatenate([joint_loc, joint_rot_aa], axis=-1)
            else:
                raise NameError('[ERROR] not valid body representation. Terminate')

            batch_betas.append(betas)
            batch_transl.append(transl)
            batch_glorot.append(pose[:,:3])
            batch_thetas.append(pose[:,3:])
            batch_jts.append(joints.reshape([-1,22*3]))
            batch_body_feature.append(body_feature)
            self.index_rec += 1
            bb += 1

            if self.index_rec == len(self.data_list):
                break
        if len(batch_betas) < batch_size:
            return None
        else:
            batch_betas = torch.cuda.FloatTensor(np.stack(batch_betas,axis=stack_dim)) #[b, t, d]
            batch_transl = torch.cuda.FloatTensor(np.stack(batch_transl,axis=stack_dim)) #[b, t, d]
            batch_glorot = torch.cuda.FloatTensor(np.stack(batch_glorot,axis=stack_dim)) #[b, t, d]
            batch_thetas = torch.cuda.FloatTensor(np.stack(batch_thetas,axis=stack_dim)) #[b, t, d]
            batch_jts = torch.cuda.FloatTensor(np.stack(batch_jts,axis=stack_dim)) #[b, t, d]
            batch_body_feature = torch.cuda.FloatTensor(np.stack(batch_body_feature,axis=stack_dim))
            return [batch_betas, batch_body_feature,
                    batch_transl, batch_glorot,batch_thetas, batch_jts]

    def get_all_data(self):
        return torch.FloatTensor(self.data_all).permute(1,0,2) #[t,b,d]





class BatchGeneratorFollowPathInCubes(object):
    def __init__(self,
                dataset_path,
                body_model_path='/home/yzhang/body_models/VPoser',
                scene_ori='ZupYf', # Z-up Y-forward, this is the default setting in our work. Otherwise we need to transform it before and after
                body_repr='ssm2_67' #['smpl_params', 'cmu_41', 'ssm2_67', 'joints']
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.dataset_path = dataset_path
        self.data_list = []
        self.body_repr = body_repr
        self.scene_ori = scene_ori
        

        self.bm_male = smplx.create(body_model_path, model_type='smplx',
                                    gender='male', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=1
                                    ).eval()
        self.bm_female = smplx.create(body_model_path, model_type='smplx',
                                    gender='female', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=1
                                    ).eval()
        self.vposer, _ = load_vposer(body_model_path+'/vposer_v1_0', vp_model='snapshot')
        self.vposer.eval()

    def params2torch(self, params, dtype = torch.float32):
        return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}

    def params2numpy(self, params):
        return {k: v.detach().cpu().numpy() for k, v in params.items() if type(v)==torch.Tensor}

    def reset(self):
        self.index_rec = 0
        random.shuffle(self.rec_list)

    def has_next_rec(self):
        if self.index_rec < len(self.rec_list):
            return True
        return False

    def get_rec_list(self, shuffle_seed=None):
        self.rec_list = sorted(glob.glob(os.path.join(self.dataset_path, 'traj_*.pkl')))
        if shuffle_seed is not None:
            random.Random(shuffle_seed).shuffle(self.rec_list)



    def xbo_to_bodydict(self, body_params):
        '''
        from what siwei gives me, to what we need to get the smplx mesh
        '''
        body_params_dict = {}
        body_params_dict['transl'] = body_params[:, :3]
        body_params_dict['global_orient'] = body_params[:, 3:6]
        body_params_dict['betas'] = body_params[:, 6:16]
        body_params_dict['body_pose_vp'] = body_params[:, 16:48]
        body_params_dict['left_hand_pose'] = body_params[:, 48:60]
        body_params_dict['right_hand_pose'] = body_params[:, 60:]
        body_params_dict['body_pose'] = self.vposer.decode(body_params[:, 16:48],
                                        output_type='aa').view(1, -1)  # tensor, [1, 63]
        return body_params_dict


    def snap_to_ground(self, xbo_dict, bm, height=0):
        ## get the body mesh and (optionally) vis in open3d
        verts = bm(**xbo_dict).vertices[0].detach().cpu().numpy()#[P, 3]
        ## snap the body mesh to the ground, and move it to a new place
        delta_z = np.array([[np.min(verts[:,-1])]])-height
        delta_xy = np.zeros((1,2))
        delta = np.concatenate([delta_xy, delta_z],axis=-1)
        xbo_dict['transl'] -= torch.FloatTensor(delta)
        return xbo_dict


    def snap_to_ground_cuda(self, xbo_dict, bm, height=0):
        ## get the body mesh and (optionally) vis in open3d
        verts = bm(**xbo_dict).vertices[0].detach()#[P, 3]
        ## snap the body mesh to the ground, and move it to a new place
        delta_z = torch.amin(verts[:,-1:],keepdim=True)-height
        delta_xy = torch.cuda.FloatTensor(1,2).zero_()
        delta = torch.cat([delta_xy, delta_z],dim=-1)
        xbo_dict['transl'] -= delta
        return xbo_dict



    def get_body_keypoints(self, xbo_dict, bm):
        bmout = bm(**xbo_dict)
        ## snap the body mesh to the ground, and move it to a new place
        markers = bmout.vertices[:,self.marker_ssm_67,:].detach().cpu().numpy()
        jts = bmout.joints[:,:22,:].detach().cpu().numpy()
        return markers, jts

    def get_bodyori_from_wpath(self, a, b):
        '''
        input: a,b #(3,) denoting starting and ending location
        '''
        z_axis = (b-a)/np.linalg.norm(b-a)
        y_axis = np.array([0,0,1])
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis/np.linalg.norm(x_axis)
        glorot_mat = np.stack([x_axis, y_axis, z_axis], axis=1)
        glorot_aa = R.from_matrix(glorot_mat).as_rotvec()
        return glorot_aa


    def next_body(self, character_file=None):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """read walking path"""
        rec_path = self.rec_list[self.index_rec]
        wpath0 = np.load(rec_path, allow_pickle=True)
        if self.scene_ori == 'YupZf':
            #rotate around x by 90, and rotate it back at the very end.
            rotmat = np.array([[1,0,0],[0,0,-1], [0,1,0]]) # rotate about x by 90deg
            wpath = [np.einsum('ij,j->i', rotmat, x) for x in wpath0]
        elif self.scene_ori == 'ZupYf':
            wpath = wpath0
        """generate a body"""
        xbo_dict = {}
        if character_file is not None:
            character_data = np.load(character_file, allow_pickle=True)
            gender = character_data['gender']
            xbo_dict['betas'] = character_data['betas']
        else:
            gender = random.choice(['female', 'male'])
            xbo_dict['betas'] = np.random.randn(1,10)

        xbo_dict['transl'] = wpath[0][None,...] #[1,3]
        xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[1])[None,...]
        xbo_dict['body_pose'] = self.vposer.decode(torch.randn(1,32), # prone to self-interpenetration
                                           output_type='aa').view(1, -1).detach().numpy()

        """snap to the ground"""
        xbo_dict = self.params2torch(xbo_dict)
        bm = self.bm_male if gender=='male' else self.bm_female
        xbo_dict = self.snap_to_ground(xbo_dict, bm, height=wpath[0][-1])

        """specify output"""
        out_dict = self.params2numpy(xbo_dict)
        out_dict['betas'] = out_dict['betas']
        out_dict['gender']=gender
        out_dict['wpath']=wpath
        out_dict['wpath_filename']=os.path.basename(rec_path)
        self.index_rec += 1

        return out_dict


    def next_body_cuda(self, character_file=None):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """read walking path"""
        rec_path = self.rec_list[self.index_rec]
        wpath = np.load(rec_path, allow_pickle=True)

        """generate a body"""
        xbo_dict = {}
        if character_file is not None:
            character_data = np.load(character_file, allow_pickle=True)
            gender = character_data['gender']
            xbo_dict['betas'] = torch.cuda.FloatTensor(character_data['betas'])
        else:
            gender = random.choice(['female'])
            xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()

        xbo_dict['transl'] = torch.cuda.FloatTensor(wpath[0][None,...]) #[1,3]
        xbo_dict['global_orient'] = torch.cuda.FloatTensor(self.get_bodyori_from_wpath(wpath[0], wpath[1])[None,...])
        xbo_dict['body_pose'] = self.vposer.decode(torch.randn(1,32), # prone to self-interpenetration
                                           output_type='aa').view(1, -1).cuda()
        """snap to the ground"""
        # xbo_dict = self.params2torch(xbo_dict)
        bm = self.bm_male if gender=='male' else self.bm_female
        bm = bm.cuda()
        xbo_dict = self.snap_to_ground_cuda(xbo_dict, bm, height=wpath[0][-1])

        """specify output"""
        out_dict = xbo_dict
        out_dict['betas']
        out_dict['gender']=gender
        out_dict['wpath']= torch.cuda.FloatTensor(np.stack(wpath))
        out_dict['wpath_filename']=os.path.basename(rec_path)
        self.index_rec += 1

        return out_dict





class BatchGeneratorReachingTarget(object):
    def __init__(self,
                dataset_path,
                body_model_path='/home/yzhang/body_models/VPoser',
                body_repr='ssm2_67' #['smpl_params', 'cmu_41', 'ssm2_67', 'joints', etc.]
                ):
        self.rec_list = list()
        self.index_rec = 0
        self.dataset_path = dataset_path
        self.data_list = []
        self.body_repr = body_repr

        self.bm_male = smplx.create(body_model_path, model_type='smplx',
                                    gender='male', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=1
                                    ).eval().cuda()
        self.bm_female = smplx.create(body_model_path, model_type='smplx',
                                    gender='female', ext='npz',
                                    num_pca_comps=12,
                                    create_global_orient=True,
                                    create_body_pose=True,
                                    create_betas=True,
                                    create_left_hand_pose=True,
                                    create_right_hand_pose=True,
                                    create_expression=True,
                                    create_jaw_pose=True,
                                    create_leye_pose=True,
                                    create_reye_pose=True,
                                    create_transl=True,
                                    batch_size=1
                                    ).eval().cuda()
        self.vposer, _ = load_vposer(body_model_path+'/vposer_v1_0', vp_model='snapshot')
        self.vposer.eval()
        self.vposer.cuda()

    def params2torch(self, params, dtype = torch.float32):
        return {k: torch.from_numpy(v).type(dtype) for k, v in params.items()}

    def params2numpy(self, params):
        return {k: v.detach().cpu().numpy() for k, v in params.items() if type(v)==torch.Tensor}

    def reset(self):
        self.index_rec = 0


    def has_next_rec(self):
        pass

    def get_rec_list(self, shuffle_seed=None):
        pass

    def snap_to_ground(self, xbo_dict, bm):
        ## get the body mesh and (optionally) vis in open3d
        verts = bm(**xbo_dict).vertices[0].detach()#[P, 3]
        ## snap the body mesh to the ground, and move it to a new place
        delta_z = torch.cuda.FloatTensor([[torch.amin(verts[:,-1])]])
        delta_xy = torch.cuda.FloatTensor(1,2).zero_()
        delta = torch.cat([delta_xy, delta_z],dim=-1)
        xbo_dict['transl'] -= delta
        return xbo_dict


    def get_bodyori_from_wpath(self, a, b):
        '''
        input: a,b #(3,) denoting starting and orientating location
        '''
        z_axis = (b-a)/torch.norm(b-a)
        y_axis = torch.cuda.FloatTensor([0,0,1])
        x_axis = torch.cross(y_axis, z_axis)
        x_axis = x_axis/torch.norm(x_axis)
        glorot_mat = torch.stack([x_axis, y_axis, z_axis], dim=1)
        glorot_aa = RotConverter.rotmat2aa(glorot_mat)[0]
        return glorot_aa


    def next_body(self, sigma=10, character_file=None):
        '''
        - get the next sample from the dataset, which is used for the target of motion planning
        '''
        """randomly specify a 2D path"""
        wpath = np.random.randn(3,3)
        # wpath = torch.cuda.FloatTensor(3, 3).normal_() #starting point, ending point, another point to initialize the body orientation
        wpath = torch.cuda.FloatTensor(wpath) #starting point, ending point, another point to initialize the body orientation
        wpath[0] = 0 #starting point
        wpath[1] = sigma*(2*torch.cuda.FloatTensor(3).uniform_()-1) #ending point
        wpath[2]=wpath[2]*sigma #initialize the body orientation
        wpath[:,-1] = 0 # proj to ground
        # wpath = np.array([[  0.        ,   0.        ,   0.        ],
        #                 [-27.91866343/2,   3.70924324/2,   0.        ],
        #                 [-13.72670315/3, -33.74804827/3,   0.        ]])

        """generate a body"""
        xbo_dict = {}
        if character_file is not None:
            character_data = np.load(character_file, allow_pickle=True)
            gender = str(character_data['gender'])
            xbo_dict['betas'] = torch.cuda.FloatTensor(character_data['betas'][:10])[None,...]
            xbo_dict['body_pose'] = torch.cuda.FloatTensor(character_data['poses'][:1,3:66])
            # xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
            #                                output_type='aa').view(1, -1)
            xbo_dict['global_orient'] = torch.cuda.FloatTensor(character_data['poses'][:1,:3])
            xbo_dict['transf_rotmat'] = torch.cuda.FloatTensor(character_data['transf_rotmat'])[None,...]
            xbo_dict['transf_transl'] = torch.cuda.FloatTensor(character_data['transf_transl'])[None,...]
        else:
            gender = random.choice(['male', 'female'])
            xbo_dict['betas'] = torch.cuda.FloatTensor(1,10).normal_()
            xbo_dict['body_pose'] = self.vposer.decode(torch.cuda.FloatTensor(1,32).normal_(), # prone to self-interpenetration
                                           output_type='aa').view(1, -1)
            xbo_dict['global_orient'] = self.get_bodyori_from_wpath(wpath[0], wpath[-1])[None,...]
            # gender = random.choice(['male'])
            # xbo_dict['betas'] = np.zeros([1,10])
        xbo_dict['transl'] = wpath[:1] #[1,3]



        """snap to the ground"""
        # xbo_dict = self.params2torch(xbo_dict)
        bm = self.bm_male if gender=='male' else self.bm_female
        xbo_dict = self.snap_to_ground(xbo_dict, bm)

        """specify output"""
        # out_dict = self.params2numpy(xbo_dict)
        xbo_dict['betas'] = xbo_dict['betas'][0]
        xbo_dict['gender']=gender
        xbo_dict['wpath']=wpath[:2]
        self.index_rec += 1

        return xbo_dict









def vis_body(bm, bparam_dict, meshgp, wpath):
    body = o3d.geometry.TriangleMesh()
    smplxout = bm(**bparam_dict)
    verts = smplxout.vertices.detach().cpu().numpy().squeeze()
    body.vertices = o3d.utility.Vector3dVector(verts)
    body.triangles = o3d.utility.Vector3iVector(bm.faces)
    body.vertex_normals = o3d.utility.Vector3dVector([])
    body.triangle_normals = o3d.utility.Vector3dVector([])
    body.compute_vertex_normals()
    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)

    ball_list = []
    for i in range(len(wpath)):
        ball = o3d.geometry.TriangleMesh.create_sphere(radius=0.5)
        ball.paint_uniform_color(np.zeros(3))
        ball.translate(wpath[i], relative=False)
        ball_list.append(ball)

    o3d.visualization.draw_geometries([body, coord, meshgp]+ball_list)




### an example of how to use it
if __name__=='__main__':
    sys.path.append(os.getcwd())
    from exp_GAMMAPrimitive.utils import config_env
    import open3d as o3d
    import trimesh
    bm_path = config_env.get_body_model_path()
    batch_gen = BatchGeneratorFollowPathInCubes(dataset_path='/mnt/hdd/tmp/slabs_with_navigation/slab008/slab_navimesh.obj_traj',
                                                body_model_path=bm_path)
    batch_gen.get_rec_list()
    for _ in range(10):
        data = batch_gen.next_body()
            # print(data['global_orient'])
            # print(data['wpath'][1])
            # print(data.keys())
        # data['transl'][:,:2]=0
        gender = data.pop('gender')
        wpath = data.pop('wpath')
        wpath_filename = data.pop('wpath_filename')
        bparam_dict =batch_gen.params2torch(data)
        bm = batch_gen.bm_male if gender=='male' else batch_gen.bm_female
        meshgp = trimesh.load("/mnt/hdd/tmp/slabs_with_navigation/slab008/slab.obj", force='mesh')
        vis_body(bm, bparam_dict, meshgp.as_open3d, wpath)


















