import torch
import os
import numpy as np
import pytorch_kinematics as pk

def quat_pos_from_transform3d(tg):
    m = tg.get_matrix()
    pos = m[:, :3, 3]
    rot = pk.matrix_to_quaternion(m[:, :3, :3])
    return pos, rot


def forward_kinematics(x_t, base_pos=[0,0,0], dist=0.052):

    d = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32

    if isinstance(x_t, np.ndarray):
        x_t = torch.from_numpy(x_t).to(dtype=dtype, device=d)

    chain = pk.build_serial_chain_from_urdf(open("assets/urdf/franka_description/robots/franka_panda.urdf").read(),"panda_hand")
    chain = chain.to(dtype=dtype, device=d)
    # print(chain.get_joint_parameter_names())

    ret = chain.forward_kinematics(x_t,end_only=False)
    # base = ret['panda_link0']
    # base_pos, _ = quat_pos_from_transform3d(base)
    # base_pos = base_pos.cpu().numpy()

    hand_tg = ret['panda_hand']
    hand_pos, _ = quat_pos_from_transform3d(hand_tg)
    hand_pos = hand_pos.cpu().numpy()

    leftfinger_tg = ret['panda_leftfinger']
    leftfinger_pos, _ = quat_pos_from_transform3d(leftfinger_tg)

    rightfinger_tg = ret['panda_rightfinger']
    rightfinger_pos, _ = quat_pos_from_transform3d(rightfinger_tg)

    pos = (rightfinger_pos+leftfinger_pos)/2
    pos = pos.cpu().numpy()
    pos = pos + np.array(base_pos)

    direction = pos - hand_pos

    pos = pos + dist*direction/(np.linalg.norm(direction,axis=1).reshape(-1,1))
    
    return pos