import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
from mpl_toolkits.mplot3d import Axes3D  


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)

DIM = 3
DOMAIN_SIZE = 2
DIVISION = 2
surface = 'sphere'
ORDER = 1
NUM_DIRECTIONS = 2


def torus2(x, y, z): #一个运算
    value = 2*y*(y**2 - 3*x**2)*(1 - z**2) + (x**2 + y**2)**2 - (9*z**2 - 1)*(1 - z**2)
    return value

def grad_torus2(x, y, z):#求梯度
    grad_x = 4 * x * (np.power(x, 2) + np.power(y, 2)) - 12 * x * y * (1 -  np.power(z, 2))
    grad_y = 2 * (1 - np.power(z, 2)) * (np.power(y, 2) - 3 * np.power(x, 2)) + 4 * y * (np.power(x, 2) + np.power(y, 2)) + 4 * np.power(y, 2) * (1 - np.power(z, 2))
    grad_z = -4 * y * z * (np.power(y, 2) - 3 * np.power(x, 2)) - 18 * (1 - np.power(z, 2)) * z + 2 * (9 * np.power(z, 2) - 1) * z
    grad = np.array([grad_x, grad_y, grad_z])
    return grad

def sphere(x, y, z):#定义一个球面
    value = np.power(x, 2) + np.power(y, 2) + np.power(z, 2) - 1
    return value

def grad_sphere(x, y, z):#球面梯度
    grad_x = 2*x
    grad_y = 2*y
    grad_z = 2*z
    grad = np.array([grad_x, grad_y, grad_z])
    return grad

def plane(x, y, z):#没弄懂是干嘛，定义了一个平面
    value = x - 1./3.#浮点
    return value

def grad_plane(x, y, z):#求这个平面的梯度
    return np.array([1., 0., 0.])


def level_set(point): #定义一个等值集
    if len(point.shape) == 1:
        x = point[0]
        y = point[1]
        z = point[2]
    else:
        x = point[:, 0]
        y = point[:, 1]
        z = point[:, 2]

    if surface == 'torus2':
        return torus2(x, y, z)
    elif surface == 'sphere':
        # print("pay your price for using global variables (value)")
        # assert 0
        return sphere(x, y, z)
    elif surface == 'plane':
        return plane(x, y, z)
    else:
        assert 0

def grad_level_set(point):#判断面的形状
    x = point[0]
    y = point[1]
    z = point[2]
    if surface == 'torus2':
        return grad_torus2(x, y, z)
    elif surface == 'sphere':
        # print("pay your price for using global variables (grad)")
        return grad_sphere(x, y, z)
    elif surface == 'plane':
        return grad_plane(x, y, z)
    else:
        assert 0


def to_id_xyz(element_id, base): #对坐标做除法运算
    id_z = element_id % base
    element_id = element_id // base
    id_y = element_id % base
    element_id = element_id // base    
    id_x = element_id % base
    element_id = element_id // base
    return id_x, id_y, id_z 

def to_id(id_x, id_y, id_z, base):
    return id_x * np.power(base, 2) + id_y * base + id_z

def get_vertices(id_x, id_y, id_z, h):#找到顶点
    vertices = []
    vertices_per_direction = 2
    for i in range(vertices_per_direction):
        for j in range(vertices_per_direction):
            for k in range(vertices_per_direction):
                vertices.append(np.array([-DOMAIN_SIZE + (id_x + i) * h, -DOMAIN_SIZE + (id_y + j) * h, -DOMAIN_SIZE + (id_z + k) * h]))
    return vertices

def breakout_id(element_id, base):
    id_x, id_y, id_z = to_id_xyz(element_id, base)
    new_ids = []
    vertices_per_direction = 2
    for i in range(vertices_per_direction):
        for j in range(vertices_per_direction):
            for k in range(vertices_per_direction):
                new_ids.append(to_id(DIVISION * id_x + i, DIVISION * id_y + j, DIVISION * id_z + k, DIVISION * base))
    return new_ids


def brute_force(base):
    ids_cut = []
    h = 2 * DOMAIN_SIZE / base
    for id_x in range(base):
        print("id_x is {}, base = {}".format(id_x, base))
        print(len(ids_cut) / np.power(base, 3))
        for id_y in range(base):
            for id_z in range(base):
                vertices = get_vertices(id_x, id_y, id_z, h)
                cut_flag, _, _ = is_cut(vertices)
                if cut_flag:
                    ids_cut.append(to_id(id_x, id_y, id_z, base))

    return ids_cut


def is_cut(vertices):
    negative_flag = False
    positive_flag = False
    for vertice in vertices:
        value = level_set(vertice)
        if value >= 0:
            positive_flag = True
        else:
            negative_flag = True
    return negative_flag and positive_flag, negative_flag, positive_flag



def generate_cut_elements():  #产生切割要素
    start_refinement_level = 5
    end_refinement_level = 7
    start_base = np.power(DIVISION, start_refinement_level)
    ids_cut = brute_force(start_base)
    total_ids = []
    total_refinement_levels = []
    total_ids.append(ids_cut)
    total_refinement_levels.append(start_refinement_level)
    print("refinement_level {}, length of inds {}".format(start_refinement_level, len(ids_cut)))
    for refinement_level in range(start_refinement_level, end_refinement_level):
        ids_cut_new = []
        base = np.power(DIVISION, refinement_level)
        h = 2 * DOMAIN_SIZE / base
        for element_id in ids_cut:
            sub_ids = breakout_id(element_id, base)
            for sub_id in sub_ids:
                sub_id_x, sub_id_y, sub_id_z = to_id_xyz(sub_id, base * DIVISION)
                cut_flag, _, _ = is_cut(get_vertices(sub_id_x, sub_id_y, sub_id_z, h / DIVISION))
                if cut_flag:
                    ids_cut_new.append(sub_id)
        ids_cut = ids_cut_new
        total_ids.append(ids_cut)
        total_refinement_levels.append(refinement_level + 1)
        print("refinement_level {}, length of inds {}".format(refinement_level + 1, len(ids_cut)))

    print("len of total_refinement_levels {}".format(len(total_refinement_levels)))
    np.savez('data/numpy/sbi/{}_cut_element_ids.npz'.format(surface), ids=total_ids, refinement_level=total_refinement_levels, allow_pickle=True)
    return total_ids, total_refinement_levels


def neighbors(element_id, base, h):
    id_xyz = to_id_xyz(element_id, base)
    faces = []
    min_id = 0
    max_id = base - 1
    for d in range(DIM):
        for r in range(NUM_DIRECTIONS):
            tmp = np.zeros(DIM)
            for i in range(DIM):
                tmp[i] = id_xyz[i]
            tmp[d] = id_xyz[d] + (2 * r - 1)
            if tmp[d] >= min_id and tmp[d] <= max_id:
                id_x, id_y, id_z = tmp
                vertices = get_vertices(id_x, id_y, id_z, h)
                cut_flag, negative_flag, positive_flag = is_cut(vertices)
                if not cut_flag and negative_flag:
                    faces.append([element_id, d*NUM_DIRECTIONS + r])
    return faces


def triangle_area(a, b, c):#三角形区域
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a))

def sbm_map_newton(point, function_value, function_gradient):
    tol = 1e-8
    res = 1.
    relax_param = 1.

    phi = function_value(point)
    grad_phi = function_gradient(point)
    target_point = np.array(point)

    step = 0
    while res > tol:
      delta1 = -phi * grad_phi / np.dot(grad_phi, grad_phi)
      delta2 = (point - target_point) - np.dot(point - target_point, grad_phi) / np.dot(grad_phi, grad_phi) * grad_phi
      target_point = target_point + relax_param * (delta1 + delta2)
      phi = function_value(target_point)
      grad_phi = function_gradient(target_point)
      res = np.absolute(phi) + np.linalg.norm(np.cross(grad_phi, (point - target_point)))
      step += 1

    # print(step)
    return target_point


def estimate_weights(shifted_q_point, d, step):
    num_boundary_points = 4 if DIM == 3 else 2
    boundary_points = np.zeros((num_boundary_points, DIM))
    for r in range(NUM_DIRECTIONS):
        for s in range(NUM_DIRECTIONS):
            boundary_points[r*NUM_DIRECTIONS + s, d] = shifted_q_point[d]
            boundary_points[r*NUM_DIRECTIONS + s, (d + 1) % DIM] = shifted_q_point[(d + 1) % DIM] + step / 2. * (2 * r - 1)
            boundary_points[r*NUM_DIRECTIONS + s, (d + 2) % DIM] = shifted_q_point[(d + 2) % DIM] + step / 2. * (2 * s - 1)

    mapped_boundary_points = np.zeros((num_boundary_points, DIM))
    for i, b_point in enumerate(boundary_points):
        mapped_boundary_points[i] = sbm_map_newton(b_point, level_set, grad_level_set)  

    mapped_q_point = sbm_map_newton(shifted_q_point, level_set, grad_level_set)

    weight = triangle_area(mapped_boundary_points[0], mapped_boundary_points[1], mapped_q_point) + \
             triangle_area(mapped_boundary_points[0], mapped_boundary_points[2], mapped_q_point) + \
             triangle_area(mapped_boundary_points[3], mapped_boundary_points[2], mapped_q_point) + \
             triangle_area(mapped_boundary_points[3], mapped_boundary_points[1], mapped_q_point)

    return mapped_q_point, weight


def process_face(face, base, h, quad_level):
    step = h / quad_level
    mapped_quad_points = []
    weights = []
    element_id, face_number = face
    id_xyz = to_id_xyz(element_id, base)
    d = face_number // NUM_DIRECTIONS
    r = face_number % NUM_DIRECTIONS
    shifted_quad_points = np.zeros((np.power(quad_level, 2), DIM))
    for i in range(quad_level):
        for j in range(quad_level):
            shifted_quad_points[i * quad_level + j, d] = -DOMAIN_SIZE + (id_xyz[d] + r) * h
            shifted_quad_points[i * quad_level + j, (d + 1) % DIM] = -DOMAIN_SIZE + id_xyz[(d + 1) % DIM]* h + step / 2. + i * step
            shifted_quad_points[i * quad_level + j, (d + 2) % DIM] = -DOMAIN_SIZE + id_xyz[(d + 2) % DIM]* h + step / 2. + j * step

    for shifted_q_point in shifted_quad_points:
        mapped_quad_point, weight = estimate_weights(shifted_q_point, d, step)
        mapped_quad_points.append(mapped_quad_point)
        weights.append(weight)

    return mapped_quad_points, weights


def compute_qw(quad_levels=[3], mesh_index=2, name='sbi_tests'):
    data = np.load('data/numpy/sbi/{}_cut_element_ids.npz'.format(surface), allow_pickle=True)
    total_ids = data['ids']
    total_refinement_levels = data['refinement_level']

    ids_cut = total_ids[mesh_index]
    refinement_level = total_refinement_levels[mesh_index]
    base = np.power(DIVISION, refinement_level)
    h = 2 * DOMAIN_SIZE / base
    print("\nrefinement_level is {} with h being {}, number of elements cut is {}".format(refinement_level, h, len(ids_cut)))

    faces = []
    for ele in range(0, len(ids_cut)):
        element_id = ids_cut[ele]
        faces += neighbors(element_id, base, h)

    for quad_level in quad_levels:
        mapped_quad_points = []
        weights = []
        ground_truth = 4 * np.pi

        for i, f in enumerate(faces):
            mapped_quad_points_f, weights_f = process_face(faces[i], base, h, quad_level)
            mapped_quad_points += mapped_quad_points_f
            weights += weights_f
            if i % 100 == 0:
                print("Progress {:.5f}%, weights {:.5f}, and gt is {:.5f}".format((i + 1)/len(faces)*100, np.sum(np.array(weights)), ground_truth))

        print(np.sum(np.array(weights)))

        case_no = 2 if surface == 'sphere' else 3

        np.savetxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_quads.dat'.format(name, 
            case_no, mesh_index, quad_level), np.asarray(mapped_quad_points).reshape(-1, DIM))
        np.savetxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_weights.dat'.format(name, 
            case_no, mesh_index, quad_level), np.asarray(weights).reshape(-1))


def test_function_0(points):
    return 1

def test_function_1(points):
    return 4 - 3 * points[:, 0]**2 + 2 * points[:, 1]**2 - points[:, 2]**2

def convergence_tests(test_function_number=0):
    name_tests = 'sbi_tests'
    name_convergence = 'sbi_convergence'
    case_no = 2 if surface == 'sphere' else 3
    # quad_levels = np.arange(1, 4, 1)
    # mesh_indices =  np.arange(0, 3, 1)
    quad_levels = np.arange(1, 3, 1)
    mesh_indices =  np.arange(0, 2, 1)
    test_function = test_function_0 if test_function_number == 0 else test_function_1
    ground_truth = 4 * np.pi if test_function_number == 0 else 40. / 3. * np.pi

    cache = False
    if not cache:
        for mesh_index in mesh_indices:
            compute_qw(quad_levels, mesh_index, name_tests)

    mesh = []
    for mesh_index in mesh_indices:
        data = np.load('data/numpy/sbi/{}_cut_element_ids.npz'.format(surface), allow_pickle=True)
        total_ids = data['ids']
        total_refinement_levels = data['refinement_level']
        # ids_cut = total_ids[mesh_index]
        refinement_level = total_refinement_levels[mesh_index]
        base = np.power(DIVISION, refinement_level)
        h = 2 * DOMAIN_SIZE / base * np.sqrt(3)
        mesh.append(h)

    errors = []
    for i, quad_level in enumerate(quad_levels):
        errors.append([])
        for j, mesh_index in enumerate(mesh_indices):
            mapped_quad_points = np.loadtxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_quads.dat'.format(name_tests, 
                case_no, mesh_index, quad_level))
            weights = np.loadtxt('data/dat/{}/sbi_case_{}_mesh_index_{}_quad_level_{}_weights.dat'.format(name_tests, 
                case_no, mesh_index, quad_level))
            values = test_function(mapped_quad_points)
            integral = np.sum(weights * values)
            relative_error = np.absolute((integral - ground_truth) / ground_truth)
            print("num quad points {}, quad_level {}, mesh_index {}, integral {}, ground_truth {}, relative error {}".format(len(weights), 
                 quad_level, mesh_index, integral, ground_truth, relative_error))
            errors[i].append(relative_error)
        convergence_array = np.concatenate((np.asarray(mesh).reshape(-1, 1), np.asarray(errors[i]).reshape(-1, 1)), axis=1)
        np.savetxt('data/dat/{}/case_{}_quad_level_{}.dat'.format(name_convergence, test_function_number, quad_level), convergence_array)

    fig = plt.figure()
    ax = fig.gca()
    for i, quad_level in enumerate(quad_levels):
        ax.plot(mesh, errors[i], linestyle='--', marker='o', label='# quad points per face {}x{}={}'.format(i + 1, i + 1, (i + 1)*(i + 1)))
 
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', prop={'size': 12})
    ax.tick_params(labelsize=14)
    ax.set_xlabel('mesh size', fontsize=14)
    ax.set_ylabel('relative error', fontsize=14)
    # fig.savefig(args.root_path + '/images/linear/L.png', bbox_inches='tight')

    print(np.log(errors[0][0]/errors[0][1]) / np.log(mesh[0]/mesh[1]))


if __name__ == '__main__':
    generate_cut_elements()
    # compute_qw()
    # convergence_tests()
    # plt.show()
