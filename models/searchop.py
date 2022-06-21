import numpy as np
import heapq
import copy
import pdb
from scipy.spatial.transform import Rotation as R



class MinHeap(object):
    def __init__(self):
        self.data = []

    def push(self, node):
        heapq.heappush(self.data, node)

    def pop(self):
        try:
            node = heapq.heappop(self.data)
        except IndexError as e:
            node=None
        return node

    def clear(self):
        self.data.clear()

    def is_empty(self):
        return True if len(self.data)==0 else False

    def deepcopy(self):
        return copy.deepcopy(self)

    def len(self):
        return len(self.data)




'''
in this script, we implement a tree of motion primitives.
MotionPrimitiveTree = MPT
From starting pose, we first generate different motion primitives as the roots, e.g. MP1, MP2, or MP3...

|-- MP1
    |-- MP1,1
    |-- MP1,2
    ...
|-- MP2
    |-- MP2,1
    |-- MP2,2
    ...
|-- MP3
    |-- MP3,1
    |-- MP3,2
    ...
'''

class MPTNode(object):
    def __init__(self, gender, betas, transf_rotmat, transf_transl, pelvis_loc, joints,
                marker, marker_proj, smplx_params, mp_latent=None, mp_type='2-frame', timestamp=-1,
                curr_target_wpath=None):
        '''
        A MPT node contains (data, parent, children list, quality)
        '''

        self.data = {}
        self.data['gender'] = gender
        self.data['betas'] = betas
        self.data['transf_rotmat'] = transf_rotmat #amass2world
        self.data['transf_transl'] = transf_transl #amass2world
        self.data['mp_type'] = mp_type
        self.data['markers'] = marker
        self.data['markers_proj'] = marker_proj
        self.data['pelvis_loc'] = pelvis_loc
        self.data['joints'] = joints
        self.data['smplx_params'] = smplx_params
        self.data['mp_latent'] = mp_latent
        self.data['timestamp'] = timestamp
        self.data['curr_target_wpath']= curr_target_wpath #(pt_idx, pt_loc)

        self.parent = None
        self.children = []
        self.quality = 0
        self.to_target_dist = 1e6
        self.motion_naturalness = 1e6
        self.q_me = 0

    def __lt__(self, other):
        '''
        note that this definition is to flip the order in the python heapq (a min heap)
        '''
        return self.quality < other.quality


    def add_child(self, child):
        '''
        child - MPTNode
        '''
        if child.quality != 0:
            child.parent = self
            self.children.append(child)
        # else:
        #     # print('[INFO searchop] cannot add low-quality children. Do nothing.')
        #     pass

    def set_parent(self, parent):
        '''
        parent - MPTNode
        '''
        if self.quality != 0:
            self.parent = parent
            return True
        else:
            return False




    def evaluate_quality_soft_contact_wpath(self,
                        terrian_rotmat=np.eye(3),
                        terrian_transl=np.zeros(3),
                        wpath=None):
        '''
        - The evaluation is based on body ground contact, in the local coordinate of terrian
        - rotmat and transl of terrian is from its local to world
        - target_node is used for planning
        - start_node is also for planning, but mainly used in A* to calculate the path cost from start to current. Set to None
        '''
        terrian_transl[-1] = wpath[0][-1]
        #----transform markers to the world coordinate
        Y_l = self.data['markers_proj'].reshape((-1, 67, 3)) #[t,p,3]
        Y_w = np.einsum('ij,tpj->tpi', self.data['transf_rotmat'][0], Y_l)+self.data['transf_transl']
        #----transform markers to the local terrian coordinate
        Y_wr = np.einsum('ij, tpj->tpi', terrian_rotmat.T, Y_w-terrian_transl[None,None,...])

        #----select motion index of proper contact with the ground
        Y_wz = Y_wr[:,:,-1] #[t, P]
        Y_w_speed = np.linalg.norm(Y_w[1:]-Y_w[:-1], axis=-1)*40 #[t=9,P=67]

        '''evaluate contact soft'''
        self.dist2g = dist2gp = max(np.abs(Y_wz.min())-0.05, 0)
        self.dist2skat = dist2skat = max(np.abs(Y_w_speed.min())-0.075,0)
        q_contact = np.exp(-dist2gp) * np.exp(-dist2skat )

        '''evaluate the distance to the final target'''
        R0 = self.data['transf_rotmat']
        T0 = self.data['transf_transl']
        target_wpath = wpath[-1]
        target_wpath_l = np.einsum('ij,j->i', R0[0].T, target_wpath-T0[0,0])[:2]
        self.dist2target=dist2target=np.linalg.norm(target_wpath_l-self.data['pelvis_loc'][-1,:2])
        q_2target = np.exp(-dist2target)

        '''evaluate facing orientation'''
        joints = self.data['joints']
        joints_end = joints[-1] #[p,3]
        x_axis = joints_end[2,:] - joints_end[1,:]
        x_axis[-1] = 0
        x_axis = x_axis / np.linalg.norm(x_axis,axis=-1,keepdims=True)
        z_axis = np.array([0,0,1])
        y_axis = np.cross(z_axis, x_axis)
        b_ori = y_axis[:2]
        t_ori = target_wpath_l[:2]-joints_end[0,:2]
        t_ori = t_ori/np.linalg.norm(t_ori, axis=-1, keepdims=True)
        dist2ori = 1-np.einsum('i,i->', t_ori, b_ori)

        curr_target_wpath = self.data['curr_target_wpath'][1]
        curr_target_wpath_l = np.einsum('ij,j->i', R0[0].T, curr_target_wpath-T0[0,0])[:2]
        self.dist2target_curr=dist2target_curr=np.linalg.norm(curr_target_wpath_l-self.data['pelvis_loc'][-1,:2])
        self.quality = dist2gp+dist2skat + 0.1*dist2ori + 0.1*dist2target_curr


    def evaluate_quality_hard_contact_wpath(self,
                        terrian_rotmat=np.eye(3),
                        terrian_transl=np.zeros(3),
                        wpath=None):
        '''
        - The evaluation is based on body ground contact, in the local coordinate of terrian
        - rotmat and transl of terrian is from its local to world
        - target_node is used for planning
        - start_node is also for planning, but mainly used in A* to calculate the path cost from start to current. Set to None
        '''
        terrian_transl[-1] = wpath[0][-1]
        #----transform markers to the world coordinate
        Y_l = self.data['markers_proj'].reshape((-1, 67, 3)) #[t,p,3]
        Y_w = np.einsum('ij,tpj->tpi', self.data['transf_rotmat'][0], Y_l)+self.data['transf_transl']
        #----transform markers to the local terrian coordinate
        Y_wr = np.einsum('ij, tpj->tpi', terrian_rotmat.T, Y_w-terrian_transl[None,None,...])

        #----select motion index of proper contact with the ground
        Y_wz = Y_wr[:,:,-1] #[t, P]
        Y_w_speed = np.linalg.norm(Y_w[1:]-Y_w[:-1], axis=-1)*40 #[t=9,P=67]

        '''evaluate contact soft'''
        # self.dist2g = dist2gp = max(np.abs(Y_wz.min())-0.05, 0)
        # self.dist2skat = dist2skat = max(np.abs(Y_w_speed.min())-0.075,0)
        if np.abs(Y_wz.min())<=0.05 and Y_w_speed.min()<=0.075:
            q_contact = 1
        else:
            q_contact = 0

        '''evaluate the distance to the final target'''
        R0 = self.data['transf_rotmat']
        T0 = self.data['transf_transl']
        target_wpath = wpath[-1]
        target_wpath_l = np.einsum('ij,j->i', R0[0].T, target_wpath-T0[0,0])[:2]
        self.dist2target=dist2target=np.linalg.norm(target_wpath_l-self.data['pelvis_loc'][-1,:2])
        q_2target = np.exp(-dist2target)

        '''evaluate facing orientation'''
        joints = self.data['joints']
        joints_end = joints[-1] #[p,3]
        x_axis = joints_end[2,:] - joints_end[1,:]
        x_axis[-1] = 0
        x_axis = x_axis / np.linalg.norm(x_axis,axis=-1,keepdims=True)
        z_axis = np.array([0,0,1])
        y_axis = np.cross(z_axis, x_axis)
        b_ori = y_axis[:2]
        t_ori = target_wpath_l[:2]-joints_end[0,:2]
        t_ori = t_ori/np.linalg.norm(t_ori, axis=-1, keepdims=True)
        dist2ori = 1-np.einsum('i,i->', t_ori, b_ori)

        curr_target_wpath = self.data['curr_target_wpath'][1]
        curr_target_wpath_l = np.einsum('ij,j->i', R0[0].T, curr_target_wpath-T0[0,0])[:2]
        self.dist2target_curr=dist2target_curr=np.linalg.norm(curr_target_wpath_l-self.data['pelvis_loc'][-1,:2])
        self.quality = q_contact*(0.1*dist2ori + 0.1*dist2target_curr)








