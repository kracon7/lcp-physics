import os
import numpy as np
import scipy.spatial as spatial
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit 

def random_action(particle_pos, particle_radius, hand_radius):
	'''
	random sample action (starting position and pushing direction) for pushing
	Input
		particle_pos - (N, 2) array, center position of object particles
		particle_radius - float, radius
		hand_radius - float, radius
	'''
	polygon, polygon_coord, normals = build_mesh(particle_pos, particle_radius)
	


def overlap_check(pts1, r1, pts2, r2):
	'''
	check if circles with center pts1 and radius r1 has overelap with circles 
	with center pts2 and radius r2
	'''
	point_tree = spatial.cKDTree(pts1)
    neighbors_list = point_tree.query_ball_point(pts2, r1 + r2)
    if len(neighbors_list) > 0:
    	return True
    else:
    	return False

def build_mesh(points, voxel_size):
	tri = Delaunay(points)
	triangles = clean_edges(points, tri.simplices, 4*voxel_size)
	edges, edge_in_tri_idx = find_boundary_edge(triangles)
	polygon = construct_polygon(edges, points, triangles, edge_in_tri_idx)
	smooth_polygon, smooth_points = corner_cutting(polygon, points, 3)
	normals = compute_polygon_normal(smooth_polygon, smooth_points)

	polygon = smooth_polygon
	polygon_coord = smooth_points
	normals = normals

	# # visualize the mesh and normal
	# fig, ax = plt.subplots(1,1)
	# plt.triplot(points[:,0], points[:,1], triangles)
	# plt.plot(points[:,0], points[:,1], 'o')
	# num_vertices = smooth_polygon.shape[0]
	# for i in range(num_vertices):
	#     pt_coord = smooth_points[smooth_polygon[i, 0]]
	#     plt.plot([pt_coord[0], pt_coord[0] + 3*normals[i,0]], 
	#                [pt_coord[1], pt_coord[1] + 3*normals[i,1]], color='r')
	# plt.plot(smooth_points[:, 0], smooth_points[:,1], '+')
	# plt.show()

	return polygon, polygon_coord, normals

def clean_edges(points, triangles, threshold=0.05):
    '''
    remove triangles with edge longer than threshold
    '''
    cleaned = []
    for t in triangles:
        p1, p2, p3 = points[t[0]], points[t[1]], points[t[2]]
        l1 = np.linalg.norm(p1-p2)
        l2 = np.linalg.norm(p1-p3)
        l3 = np.linalg.norm(p3-p2)
        if (l1 < threshold) & (l2 < threshold) & (l3 < threshold):
            cleaned.append(t)
            
    cleaned = np.stack(cleaned)
    return cleaned

def triangle_to_edge_set(triangles):
    '''
    extract edges for each triangles, no duplicated count
    Input
        triangles -- vertex index of each triangle (N, 3)
    Output
        edges -- the set of edges
        tri_idx -- corresponding index
    '''
    edges = np.empty((0,2))
    tri_idx = []
    for i, t in enumerate(triangles):
        edge_candidates = [np.array([t[0], t[1]]), np.array([t[1], t[0]]),
                           np.array([t[0], t[2]]), np.array([t[2], t[0]]),
                           np.array([t[1], t[2]]), np.array([t[2], t[1]])]
        for e in edge_candidates:
            # if e already existes in edges
            if ((edges[:,0]==e[0]) & (edges[:,1]==e[1])).any() or \
                ((edges[:,0]==e[1]) & (edges[:,1]==e[0])).any():
                continue
            else:
                edges = np.concatenate([edges, e.reshape(1,2)], axis=0)
                tri_idx.append(i)
            
    return edges, tri_idx

def find_boundary_edge(triangles):
    '''
    Extract the boundary edges, the boundary edges only appears in the triangles once
    This function first extract edges for each triangles, non-boundary edges will appear twice
    Then use traingle_to_edge_set() to extract non-duplicated edges
    Input
        triangles -- vertex index of each triangle (N, 3)
    Output
        edges_boundary -- boundary edges
        idx -- corresponding triangle index
    '''
    # extract all edges from each triangles, with duplicated count
    edges_dup = []
    for t in triangles:
        edges_dup += [np.array([t[0], t[1]]), 
                  np.array([t[0], t[2]]),
                  np.array([t[1], t[2]])]
    edges_dup = np.stack(edges_dup)
    
    # extract non-duplicated edges
    edges_set, edge_in_tri_idx = triangle_to_edge_set(triangles)
    
    edges_boundary, idx = [], []
    for i, e in enumerate(edges_set):
        # check how many times e appear in edge_dup
        N1 = np.where((edges_dup[:,0]==e[0]) & (edges_dup[:,1]==e[1]))[0].shape[0]
        N2 = np.where((edges_dup[:,0]==e[1]) & (edges_dup[:,1]==e[0]))[0].shape[0]
        N = N1 + N2
        if N == 1:
            edges_boundary.append(e)
            idx.append(edge_in_tri_idx[i])
        elif N > 1:
            continue
        else:
            print('something went wrong with the edge extraction!')
    
    edges_boundary = np.stack(edges_boundary)

    return edges_boundary, idx

def construct_polygon(edges_boundary, points, triangles, edge_in_tri_idx ):
    '''
    Construct the polygon from boundary edges
    Input
        edges_boundary -- (N, 2)
        triangles -- vertex index of each triangle (N, 3)
        edge_in_tri_idx -- corresponding triangle index
    Output
        polygon -- point index of each edge (M, 2)
    '''
    # boundary edge number
    N = edges_boundary.shape[0]
    # initialize the first edge
    polygon = [[edges_boundary[0,0], edges_boundary[0,1]]]
    for i in range(1, N):
        # current edge
        e = [polygon[-1][0], polygon[-1][1]]
        # edges contains the open-end vertex in current edge
        idx = np.where(edges_boundary==e[1])
        assert idx[0].shape[0] == 2
        
        p1 = e[-1]
        # if the first row is the original edge
        if (edges_boundary[idx[0][0]]==e[0]).any():
            p2 = edges_boundary[idx[0][1], 1-idx[1][1]]
        else:
            p2 = edges_boundary[idx[0][0], 1-idx[1][0]]
        polygon.append([p1, p2])
            
    polygon = np.array(polygon).astype('int')
    
    # check chirality of the polygon edges
    edge = polygon[0]
    tri = triangles[edge_in_tri_idx[0]]
    p1, p2 = edge[0], edge[1]
    p3 = tri[(tri != p1) & (tri != p2)]
    vec1 = points[p2] - points[p1]
    vec2 = points[p3] - points[p1]
    # inside lies on the left side of the edge vector, right hand rule
    if np.cross(vec1, vec2) < 0:
        polygon = np.flip(np.flip(polygon, axis=0), axis=1)
    
    return polygon

def corner_cutting(polygon, points, num_iter=2):
    '''
    smooth polygon using chaikin's corner cutting algorithm
    Input
        polygon -- edges of polygon (m, 2)
        points -- coordinates of all the original vertices (N, 2)
    '''
    # build new polygon and points
    new_polygon, new_points = [], []
    for i, edge in enumerate(polygon):
        if i != polygon.shape[0] - 1:
            new_polygon.append([i, i+1])
        else:
            new_polygon.append([i, 0])
        
        new_points.append(points[polygon[i,0]])
    
    new_polygon = np.array(new_polygon)
    new_points = np.stack(new_points)
    
    for _ in range(num_iter):
        smooth_points= []
        
        for i in range(new_polygon.shape[0]):
            p1, p2 = new_polygon[i]
            
            q1 = 4/5 * new_points[p1] + 1/5 * new_points[p2]
            q2 = 1/5 * new_points[p1] + 4/5 * new_points[p2]
            
            smooth_points += [q1, q2]
        
        # re-assign new polygon point index
        new_polygon = []
        for i in range(len(smooth_points)):
            new_polygon.append([i, i+1])
        new_polygon[-1][1] = 0
        new_polygon = np.array(new_polygon)
#         print(new_polygon)
            
        # re-assign new point coordinates    
        new_points = np.stack(smooth_points).copy()
    
    return new_polygon, new_points


def compute_polygon_normal(smooth_polygon, smooth_points, num_ave=3):
    '''
    Compute the normal directions of the vertices on the boundary of a polygon
    Input
        smooth_polygon -- the smoothed edges of the polygon ordered by right hand rule (m,2)
        smooth_points -- the 2D coordinates of the points (N, 2)
    '''
    num_vertices = smooth_points.shape[0]
    
    def line_objective(x, a, b):
        return a * x + b
    
    # compute the normal direction that is on the same side as exterior vectors
    # line fit the points on both sides to get the normal direction
    normals = []
    expand_edges = np.concatenate([smooth_polygon, smooth_polygon[:num_ave]])
    print(num_vertices, expand_edges.shape)
    for i in range(num_vertices):
        # find the point coordinates to perform the line fit
        vt_idx = [j for j in range(i-num_ave, i+num_ave+1)]
        coord = smooth_points[expand_edges[[vt_idx],0]].reshape(-1,2)
        popt, _ = curve_fit(line_objective, coord[:,0], coord[:,1])
        a, b = popt
        norm = np.array([a, -1])
        
        # check whether direction points to exterior
        vec = smooth_points[smooth_polygon[i, 1]] - smooth_points[smooth_polygon[i, 0]]
        if np.cross(norm, vec) < 0:
            norm = -norm
            
        normals.append(norm / np.linalg.norm(norm))
        
    normals = np.stack(normals)
    
    return normals