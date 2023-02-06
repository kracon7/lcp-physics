import os
import argparse
import torch
import numpy as np
import taichi as ti
from lcp_physics.physics.hidden_state_mapping import HiddenStateMapping
from lcp_physics.physics.grasp_n_rotate_simulator import GraspNRotateSimulator

ti.init(arch=ti.cpu)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def render(gui, np_pos, wx_min, wx_max, wy_min, wy_max, radius, resol_x, resol_y, fname=None):
    if np_pos.shape[1] == 3:
        np_pos = np_pos[:,1:]
    np_pos = (np_pos - np.array([wx_min, wy_min])) / \
             (np.array([wx_max - wx_min, wy_max - wy_min]))

    r = radius * resol_x / (wx_max-wx_min)
    gui.circles(np_pos, color=0xffffff, radius=r)
    gui.show(fname)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='System ID on block object model')
    parser.add_argument("--param_file", type=str, default='block_object_param.yaml')
    parser.add_argument("--data_dir", type=str, default='block_object_param_1')
    args = parser.parse_args()

    param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
    sim = GraspNRotateSimulator(param_file)

    data_dir = os.path.join(ROOT, 'data', args.data_dir)
    mass = np.loadtxt(os.path.join(data_dir, "composite_mass.txt"))
    friction = np.loadtxt(os.path.join(data_dir, "composite_friction.txt"))
    mass_mapping = np.loadtxt(os.path.join(data_dir, "mass_mapping.txt")).astype("int")
    friction_mapping = np.loadtxt(os.path.join(data_dir, "friction_mapping.txt")).astype("int")
    
    hidden_state_mapping = HiddenStateMapping(sim)
    hidden_state = hidden_state_mapping.map_to_hidden_state(mass, mass_mapping,
                                                    friction, friction_mapping)

    u, u_idx = {}, [0, 10, 20]
    batch_speed = np.loadtxt(os.path.join(data_dir, 'batch_speed.txt'))
    for i in u_idx:
        external_ft = np.loadtxt(os.path.join(data_dir, "ft_%d.txt"%i))
        u[i] = external_ft

    mass = torch.from_numpy(mass).double()
    friction = torch.from_numpy(friction).double()

    for i in u_idx:
        rotation_center = sim.particle_pos0[i]
        rotation_speed = torch.tensor([batch_speed[i, 1]]).double()
        init_vel = sim.get_init_vel(rotation_center, rotation_speed)
        sim.initialize(mass, mass_mapping, friction, friction_mapping, i, u[i], init_vel)
        world = sim.make_world()
        run_time = u[i][-1, 0] - u[i][0, 0]
        positions = sim.positions_run_world(world, run_time=run_time, screen=None, recorder=None)

    resol_x, resol_y = 800, 800
    gui = ti.GUI("Particle positions", (resol_x, resol_y))

    for t, p in positions.items():
        p = p.detach().numpy()
        render(gui, p, 0.05, 0.65, -0.3, 0.3, 0.0125, resol_x, resol_y)