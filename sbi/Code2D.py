import sys
import numpy as np
import matplotlib as plt
import copy


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)

DIM = 2
DOMAIN_SIZE = 2 # 实际上我们定义的区域是[-2, 2]x[-2, 2]的正方形（2D情况）
DIVISION = 2
ORDER = 1
NUM_DIRECTIONS = 2

def level_set(point, ls_fn):#判断
    x = point[0]
    y = point[1]
    return ls_fn(x, y)


def grad_level_set(point, ls_grad_fn):#求等值集的梯度
    x = point[0]
    y = point[1]
    return ls_grad_fn(x, y)


def to_id_xy(element_id, base):
    
    id_y = element_id % base
    element_id = element_id // base    
    id_x = element_id % base
    element_id = element_id // base
    return id_x, id_y

def to_id(id_x, id_y, base): 
    
    return id_x * base + id_y


def get_element_sub_ids(element_id, base):
    
    id_x, id_y = to_id_xy(element_id, base)
    new_ids = []
    vertices_per_direction = 2
    for i in range(vertices_per_direction):
        for j in range(vertices_per_direction):
            new_ids.append(to_id(DIVISION * id_x + i, DIVISION * id_y + j, DIVISION * base))
    return new_ids


def get_vertices(id_x, id_y, h):
    
    vertices = []
    vertices_per_direction = 2
    for i in range(vertices_per_direction):
        for j in range(vertices_per_direction):
            vertices.append(np.array([-DOMAIN_SIZE + (id_x + i) * h, -DOMAIN_SIZE + (id_y + j) * h]))
    vertices = np.stack(vertices)
    return vertices


def is_cut(vertices, ls_fn):
    
    negative_flag = False
    positive_flag = False
    for vertice in vertices:#对顶点集的每个顶点做循环
        value = level_set(vertice, ls_fn)
        if value >= 0: # 用户定义的函数大于等于零，则点在区域内
            positive_flag = True
        else: # 反之，点在区域外
            negative_flag = True
    return negative_flag and positive_flag, negative_flag, positive_flag#第一个就是指这个单元既有在外面的点又有在里面的点，顾该单元是被函数线穿过的单元


def brute_force(base, ls_fn):
    
    ids_cut = []
    h = 2 * DOMAIN_SIZE / base
    for id_x in range(base):
        print(f"id_x is {id_x}, base = {base}")
        print(len(ids_cut) / np.power(base, 2))
        for id_y in range(base):
            vertices = get_vertices(id_x, id_y, h)
            cut_flag, _, _ = is_cut(vertices, ls_fn)
            if cut_flag:
                ids_cut.append(to_id(id_x, id_y, base))
    return ids_cut


def neighbors(element_id, base, h, ls_fn):
    """给定被曲线穿过的单元，如何得到我们想要的surrogate boundary?
    
    Args:
        element_id (int): global index
        base (int): number of elements per axis
        h (float): element size
    
    Returns:
        list: list of sides that form the surrogate boundary
    """
    id_xy = to_id_xy(element_id, base)
    sides = []
    min_id = 0
    max_id = base - 1
    for d in range(DIM):#DIM应该是dimension，数值为2，对应x, y两个方向
        for r in range(NUM_DIRECTIONS):#NUM_DIRECTION = 2, 对应0和1，r为0就是原坐标-1的坐标对应的格子，r为1就是原坐标+1对应的坐标的格子
            tmp = np.array(id_xy)
            tmp[d] = id_xy[d] + (2 * r - 1)#这里是对该小格子的相邻六个小格子进行处理，即上下左右六个小格子
            if tmp[d] >= min_id and tmp[d] <= max_id:
                id_x, id_y = tmp
                vertices = get_vertices(id_x, id_y, h)#get_vertices会对应到x, y, z轴的坐标值
                cut_flag, negative_flag, positive_flag = is_cut(vertices, ls_fn)
                if not cut_flag and negative_flag:
                    sides.append([element_id, d*NUM_DIRECTIONS + r])
    return sides#sides是个二维列表，存储的是element_id和该element_id对应的形成surrogate boundary的边


def segment_distance(a, b):
    """TODO: 2维情况应该怎么处理？
    """
    dot1 = np.array(a)
    dot2 = np.array(b)

    return np.linalg.norm(dot1 - dot2)


def sbm_map_newton(point, function_value, function_gradient, ls_fn, ls_grad_fn):
    """核心算法,对应JCP论文里的章节为"3.2. Considerations for maps"
    JCP论文地址：https://doi.org/10.1016/j.jcp.2021.110360
    """
    tol = 1e-8 #指1乘以10的-8次方
    res = 1.
    relax_param = 1.#松弛参数（？
    #这里的Newton method三个重要参数如下
    phi = function_value(point, ls_fn)#把估计值带入函数得到phi
    grad_phi = function_gradient(point, ls_grad_fn)#把估计值带入函数的导数得到grad_phi
    target_point = np.array(point)##point看成初始估计，并且会不断向误差较小的方向进行调整

    step = 0
    while res > tol:#不断缩小误差
      delta1 = -phi * grad_phi / np.dot(grad_phi, grad_phi)#dot用来计算向量的点积和矩阵的乘法
      delta2 = (point - target_point) - np.dot(point - target_point, grad_phi) / np.dot(grad_phi, grad_phi) * grad_phi
      target_point = target_point + relax_param * (delta1 + delta2)
      phi = function_value(target_point, ls_fn)
      grad_phi = function_gradient(target_point, ls_grad_fn)
      res = np.absolute(phi) + np.linalg.norm(np.cross(grad_phi, (point - target_point)))
      step += 1

    # print(step)
    return target_point


def estimate_weights(shifted_q_point, d, step, ls_fn, ls_grad_fn):
    """辅助计算
    """
    num_boundary_points = 2 #DIM
    boundary_points = np.zeros((num_boundary_points, DIM))
    for r in range(NUM_DIRECTIONS): #取0，1
            boundary_points[r, d] = shifted_q_point[d]
            boundary_points[r, (d + 1) % DIM] = shifted_q_point[(d + 1) % DIM] + step / 2. * (2 * r - 1)
            #计算出每个shifted_q_point所对应的线段的2个顶点

    mapped_boundary_points = np.zeros((num_boundary_points, DIM))
    for i, b_point in enumerate(boundary_points):#把每个boundary_point映射到函数上（通过牛顿法来求解该mapped点的坐标）
        mapped_boundary_points[i] = sbm_map_newton(b_point, level_set, grad_level_set, ls_fn, ls_grad_fn)#用牛顿法把映射后对应的坐标表示出来  
    #下面一行表示将shifted_q_point映射到曲面函数上
    mapped_q_point = sbm_map_newton(shifted_q_point, level_set, grad_level_set, ls_fn, ls_grad_fn)

    weight = segment_distance(mapped_boundary_points[0], mapped_q_point) + segment_distance(mapped_boundary_points[1], mapped_q_point) 
    #weight值就是两个线段长度之和
    return mapped_q_point, weight


def process_sides(side, base, h, quad_level, ls_fn, ls_grad_fn):
    """对于每一个边，计算在边上对应的积分点和权重，quad_level是指该side被分割的程度
    """
    step = h / quad_level#把side的边长进行更近一步的划分
    mapped_quad_points = []#用这个列表存储映射后在曲面上的坐标点
    weights = []#用这个列表来储存权重，即两段线段的长度
    element_id, side_number = side
    id_xy = to_id_xy(element_id, base)
    d = side_number // NUM_DIRECTIONS #NUM_DIRECTION = 2，还原参数d, 可以取0，1
    r = side_number % NUM_DIRECTIONS #还原参数r，可以取0，1
    shifted_quad_points = np.zeros((np.power(quad_level, 1), DIM)) #返回一个用0填充的数组
    for i in range(quad_level):
            shifted_quad_points[i , d] = -DOMAIN_SIZE + (id_xy[d] + r) * h
            shifted_quad_points[i , (d + 1) % DIM] = -DOMAIN_SIZE + id_xy[(d + 1) % DIM]* h + step / 2. + i * step
    #用上面这个循环算出了该面上的积分点的坐标值
    for shifted_q_point in shifted_quad_points:#对每一个shifted_quad_points作处理，计算对应的权重长度值
        mapped_quad_point, weight = estimate_weights(shifted_q_point, d, step, ls_fn, ls_grad_fn)
        mapped_quad_points.append(mapped_quad_point)
        weights.append(weight)
    return mapped_quad_points, weights


def generate_cut_elements(ls_fn):
    
    start_refinement_level = 5
    end_refinement_level = 7
    start_base = np.power(DIVISION, start_refinement_level) 
    ids_cut = brute_force(start_base, ls_fn)
    total_ids = []
    total_refinement_levels = []
    total_ids.append(ids_cut)
    total_refinement_levels.append(start_refinement_level)
    print(f"refinement_level {start_refinement_level}, length of inds {len(ids_cut)}") 

    for refinement_level in range(start_refinement_level, end_refinement_level):
        ids_cut_new = []
        base = np.power(DIVISION, refinement_level)
        h = 2 * DOMAIN_SIZE / base
        for element_id in ids_cut:
            sub_ids = get_element_sub_ids(element_id, base)
            for sub_id in sub_ids:
                sub_id_x, sub_id_y = to_id_xy(sub_id, base * DIVISION)
                cut_flag, _, _ = is_cut(get_vertices(sub_id_x, sub_id_y, h / DIVISION), ls_fn)
                if cut_flag:
                    ids_cut_new.append(sub_id)
        ids_cut = ids_cut_new
        total_ids.append(ids_cut)
        total_refinement_levels.append(refinement_level + 1)
        print(f"refinement_level {refinement_level + 1}, length of inds {len(ids_cut)}")

    total_ids = np.array(total_ids, dtype=object)
    total_refinement_levels = np.array(total_refinement_levels)
    print(f"len of total_refinement_levels {len(total_refinement_levels)}")
    return total_ids, total_refinement_levels



def compute_qw(total_ids, total_refinement_levels, ls_fn, ls_grad_fn, quad_level, mesh_index):
    """第二个大的步骤，计算积分点(quadrature point)以及积分点对应的权重(weight)
    有了积分点和权重，算曲线积分就易如反掌了。
    """
    ids_cut = total_ids[mesh_index]#total_ids是一个二维列表，ids_cut是一个一维列表，存有某一个refinement_level下的筛选后的编号
    refinement_level = total_refinement_levels[mesh_index]
    base = np.power(DIVISION, refinement_level)
    h = 2 * DOMAIN_SIZE / base
    print("\nrefinement_level is {} with h being {}, number of elements cut is {}".format(refinement_level, h, len(ids_cut)))
    sides = []
    for ele in range(len(ids_cut)):
        element_id = ids_cut[ele]#这个for循环是为了遍历每一个ids_cut中的每一个编号
        sides += neighbors(element_id, base, h, ls_fn)#找到list of sides that form the surrogate boundary

    mapped_quad_points = []
    weights = []
    for i, f in enumerate(sides):#遍历每一个边
        mapped_quad_points_f, weights_f = process_sides(sides[i], base, h, quad_level, ls_fn, ls_grad_fn)
        mapped_quad_points += mapped_quad_points_f
        weights += weights_f
        if i % 100 == 0:
            print(f"Progress {(i + 1)/len(sides)*100:.5f}%, weights {np.sum(np.array(weights)):.5f}")

    mapped_quad_points = np.array(mapped_quad_points)
    weights = np.array(weights)

    return mapped_quad_points, weights
