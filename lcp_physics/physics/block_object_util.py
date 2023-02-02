import os
import yaml
import numpy as np
import plotly.graph_objects as go

'''
Shape coordinate frame
        1 1 1 1
        1 1 1 1
        1 1       ^ X
        1 1         |
        1 1 1 1     |
        1 1 1 1     |
                    |
        <-----------O
        Y

Mass mapping order
        3 3 4 4
        3 3 4 4
        2 2         ^ X
        2 2         |
        0 0 1 1     |
        0 0 1 1     |
                    |
        <-----------O
        Y
'''

class BlockObject:
    def __init__(self, param_file) -> None:
        f = open(param_file, 'r')
        self.params = yaml.load(f, Loader=yaml.Loader)
        f.close()

        shape_file = os.path.join(os.path.dirname(os.path.dirname(param_file)), 
                                    "shapes", self.params['shape'])
        self.shape_array = np.loadtxt(shape_file)
        resol = self.params['resol']                         # resolution of discretization for each block
        block_size = self.params['block_size']               # side length of the block object (in cm)
        self.voxel_size = block_size / resol                 # side length of each voxel (in cm)
        m_b = self.shape_array.shape[0] // resol             # number of blocks vertically
        n_b = self.shape_array.shape[1] // resol             # number of blocks horizontally
        self.num_particle = int(self.shape_array.sum())

        mass_idx = 0
        coord, mass_mapping = [], []
        for i in range(m_b):
            for j in range(n_b):
                block_idx = [m_b - i, j]
                array_idx = [resol * block_idx[0] - (resol // 2) - 1,
                             resol * block_idx[1] + ((resol - 1) // 2)]
                # print("i: %d, j: %d , array i: %d, j: %d, block i: %d, j: %d"%\
                #         (i, j, array_idx[0], array_idx[1], block_idx[0], block_idx[1]))
                if self.shape_array[array_idx[0], array_idx[1]] == 1:
                    for k in range(-((resol-1) // 2), resol // 2 + 1):
                        for l in range(-((resol-1) // 2), resol // 2 + 1):
                            # particle coordinate in object bottom left frame
                            coord.append([ i * block_size + k * self.voxel_size,
                                          -j * block_size + l * self.voxel_size])
                            # mass mapping of the particle
                            mass_mapping.append(mass_idx)
                    # increment the mass index by 1 for the next block
                    mass_idx += 1

        self.particle_coord = (np.array(coord) + np.array(self.params['offset']))            # particle coord in meter
        self.mass_mapping = np.array(mass_mapping).astype('int')
        self.friction_mapping = np.array(mass_mapping).astype('int')

    def plot(self):
        coord = self.particle_coord
        fig = go.Figure(
                data = go.Scatter(x=coord[:, 0], y=coord[:, 1], mode = 'markers'),
                layout=go.Layout(
                    xaxis=dict(range=[0.05, 0.65], autorange=False),
                    yaxis=dict(range=[-0.3, 0.3], autorange=False),
                    height=800, width=800
            )
        )
        fig.show()


# # -----------------------------------------     TEST    --------------------------------------------------
# ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
# block_object = BlockObject(param_file)
# coord = block_object.particle_coord
# print(block_object.mass_mapping)

# fig = go.Figure(
#         data = go.Scatter(x=coord[:, 0], y=coord[:, 1], mode = 'markers'),
#         layout=go.Layout(
#             xaxis=dict(range=[0.05, 0.65], autorange=False, title="X"),
#             yaxis=dict(range=[-0.3, 0.3], autorange=False, title="Y"),
#             height=800, width=800
#     )
# )
# fig.show()

# -----------------------------------------     Archive    --------------------------------------------------
# shape_file = os.path.join(ROOT, "shapes", "F.txt")
# shape_array = np.loadtxt(shape_file)
# params_file = open(os.path.join(ROOT, 'config', 'block_object_param.yaml'), 'r')
# params = yaml.load(params_file, Loader=yaml.Loader)
# params_file.close()
# resol = params['resol']                         # resolution of discretization for each block
# block_size = params['block_size']               # side length of the block object (in cm)
# voxel_size = block_size / resol                 # side length of each voxel (in cm)
# m_b = shape_array.shape[0] // 3                 # number of blocks vertically
# n_b = shape_array.shape[1] // 3                 # number of blocks horizontally

# num_particles = shape_array.sum()

# coord = []
# for i in range(m_b):
#     for j in range(n_b):
#         block_idx = [m_b - i, n_b - j]
#         array_idx = [resol * block_idx[0] - (resol // 2) - 1,
#                      resol * block_idx[1] - (resol // 2) - 1]
#         if shape_array[array_idx[0], array_idx[1]] == 1:
#             for k in range(-(resol // 2), resol // 2 + 1):
#                 for l in range(-(resol // 2), resol // 2 + 1):
#                     coord.append([(block_idx[0] - 1) * block_size + k * voxel_size,
#                                 -(block_idx[1] - 1) * block_size + l * voxel_size])
# coord = np.array(coord)

# fig = go.Figure(
#         data = go.Scatter(x=coord[:, 0], y=coord[:, 1], mode = 'markers'),
#         layout=go.Layout(
#             xaxis=dict(range=[-5, 55], autorange=False),
#             yaxis=dict(range=[-30, 30], autorange=False),
#             height=800, width=800
#     )
# )
# fig.show()