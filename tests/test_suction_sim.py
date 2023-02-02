import os
from lcp_physics.physics.suction_simulator import SuctionSimulator

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
param_file = os.path.join(ROOT, 'config', 'block_object_param.yaml')
sim = SuctionSimulator(param_file)