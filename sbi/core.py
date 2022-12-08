import sys
import numpy as np
import matplotlib.pyplot as plt
import copy


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)

DIM = 3
DOMAIN_SIZE = 2 # 实际上我们定义的区域是[-2, 2]x[-2, 2]x[-2, 2]这个立方体（3D情况）
DIVISION = 2
ORDER = 1
NUM_DIRECTIONS = 2


def level_set(point, ls_fn):#判断
    if len(point.shape) == 1:#用shape读取矩阵长度
        x = point[0]
        y = point[1]
        z = point[2]
    else:
        x = point[:, 0] #表示对二维数组，取所有行的第0个元素
        y = point[:, 1] #表示对二维数组，取所有行的第1个元素
        z = point[:, 2] #表示对二维数组，取所有行的第2个元素
    return ls_fn(x, y, z)#感觉可能没有用if的必要（？


def grad_level_set(point, ls_grad_fn):#求等值集的梯度
    x = point[0]
    y = point[1]
    z = point[2]
    return ls_grad_fn(x, y, z)

 
def to_id_xyz(element_id, base):
    """每个单元有一个全局编号，例如在一个3x3x3的魔方里面，element_id可以取值0,1,2,...,26
    但是也可以用一个三元数组类似于x-y-z的方法来编号，比如(0,0,1),(0,0,2),...,(2,2,2)
    这个函数将element_id映射到(id_x, id_y, id_z)
    
    Args:
        element_id (int): global index
        base (int): number of elements per axis
    
    Returns:
        tuple: 
    """
    id_z = element_id % base
    element_id = element_id // base
    id_y = element_id % base
    element_id = element_id // base    
    id_x = element_id % base
    element_id = element_id // base
    return id_x, id_y, id_z 
    #下一个函数就是这个函数的逆运算，这个函数把element_id转化为了三维坐标下的坐标

def to_id(id_x, id_y, id_z, base): 
    """对于某个给定的单元，这个函数将其编号(id_x, id_y, id_z)映射到element_id
    
    Args:
        id_x (int): x-axis index
        id_y (int): y-axis index
        id_z (int): z-axis index
        base (int): number of elements per axis
    
    Returns:
        int: global index
    """
    return id_x * np.power(base, 2) + id_y * base + id_z

    #（xiong）思考：把三维坐标 和 一维数字 做映射 的意义是什么？（为什么会想用这个方式来计算element_id映射？） 在2D环境里还需要这样做吗？


def get_vertices(id_x, id_y, id_z, h):
    """用这个函数找到顶点
    
    Args:
        id_x (int): x-axis index
        id_y (int): y-axis index
        id_z (int): z-axis index
        h (float): element size
    
    Returns:
        np.ndarray: vertices coordinates, e.g., shape is (8, 3) for 3D case
    """
    vertices = []
    vertices_per_direction = 2#沿每个方向上的顶点数是2个，即0，1 
    for i in range(vertices_per_direction):
        for j in range(vertices_per_direction):
            for k in range(vertices_per_direction):
                vertices.append(np.array([-DOMAIN_SIZE + (id_x + i) * h, -DOMAIN_SIZE + (id_y + j) * h, -DOMAIN_SIZE + (id_z + k) * h]))
    vertices = np.stack(vertices)#把id_x, y, z 对应到三个轴的坐标上去
    return vertices#一个二维列表，里面包含8个实际坐标列表


def get_element_sub_ids(element_id, base):
    """给定一个单元，编号为element_id（比如在一个4x4x4的网格中，编号为10）
    那么我们可以手动计算出它的xyz编号为(0, 2, 2), 或者可以使用 to_id_xyz(10, 4)这个函数得到。
    现在将这个4x4x4的网格加密为8x8x8，那么这个编号为10的单元将被划分为了8个小单元。
    这个函数返回的就是这个8个小单元，在新的编号系统下的全局编号。
    例如，在这个例子里，如果调用 get_element_sub_ids(10, 4), 将可以得到 [36, 37, 44, 45, 100, 101, 108, 109]
    
    Args:
        element_id (int): global index
        base (int): number of elements per axis
    
    Returns:
        list[int]: new element indices of the divided smaller elements
    """
    id_x, id_y, id_z = to_id_xyz(element_id, base)
    new_ids = []
    vertices_per_direction = 2
    for i in range(vertices_per_direction):
        for j in range(vertices_per_direction):
            for k in range(vertices_per_direction):
                new_ids.append(to_id(DIVISION * id_x + i, DIVISION * id_y + j, DIVISION * id_z + k, DIVISION * base))
    return new_ids


def brute_force(base, ls_fn):
    """用“蛮力”遍历网格中的每一个单元，判断它是否被曲面零等高线穿过
    
    Args:
        base (int): number of elements per axis
    
    Returns:
        list[int]: a list of element indices 
    """
    ids_cut = []
    h = 2 * DOMAIN_SIZE / base  
    for id_x in range(base):
        print(f"id_x is {id_x}, base = {base}")
        print(len(ids_cut) / np.power(base, 3)) #打印的这句话是什么意思？
        for id_y in range(base):
            for id_z in range(base):
                vertices = get_vertices(id_x, id_y, id_z, h)
                cut_flag, _, _ = is_cut(vertices, ls_fn)
                if cut_flag:
                    ids_cut.append(to_id(id_x, id_y, id_z, base))
    return ids_cut


def is_cut(vertices, ls_fn):
    """判断一个单元是否被我们要研究的曲面的零等高线穿过
    
    Args:
        vertices (np.ndarray): element vertex coordinates
    
    Returns:
        bool: whether the element is cut or not
    """
    negative_flag = False
    positive_flag = False
    for vertice in vertices:#对顶点集的每个顶点做循环
        value = level_set(vertice, ls_fn)
        if value >= 0: # 用户定义的函数大于等于零，则点在区域内
            positive_flag = True
        else: # 反之，点在区域外
            negative_flag = True
    return negative_flag and positive_flag, negative_flag, positive_flag#第一个就是指这个单元既有在外面的点又有在里面的点，顾该单元是被函数线穿过的单元


def neighbors(element_id, base, h, ls_fn):
    """给定被曲面穿过的单元，如何得到我们想要的surrogate boundary?
    
    Args:
        element_id (int): global index
        base (int): number of elements per axis
        h (float): element size
    
    Returns:
        list: list of faces that form the surrogate boundary
    """
    id_xyz = to_id_xyz(element_id, base)
    faces = []
    min_id = 0
    max_id = base - 1
    for d in range(DIM):#DIM应该是dimension，数值为3，对应x, y, z三个方向
        for r in range(NUM_DIRECTIONS):#NUM_DIRECTION = 2, 对应0和1，r为0就是原坐标-1的坐标对应的格子，r为1就是原坐标+1对应的坐标的格子
            tmp = np.array(id_xyz)
            tmp[d] = id_xyz[d] + (2 * r - 1)#这里是对该小格子的相邻六个小格子进行处理，即上下左右六个小格子
            if tmp[d] >= min_id and tmp[d] <= max_id:
                id_x, id_y, id_z = tmp
                vertices = get_vertices(id_x, id_y, id_z, h)#get_vertices会对应到x, y, z轴的坐标值
                cut_flag, negative_flag, positive_flag = is_cut(vertices, ls_fn)
                if not cut_flag and negative_flag:
                    faces.append([element_id, d*NUM_DIRECTIONS + r])
    return faces#faces是个二维列表，存储的是element_id和该element_id对应的形成surrogate boundary的面


def triangle_area(a, b, c):
    """TODO: 2维情况应该怎么处理？
    """
    return 0.5 * np.linalg.norm(np.cross(b - a, c - a)) #np.linalg.norm求范数，默认ord为2，np.cross求叉乘


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
    num_boundary_points = 4 #if DIM == 3 else 2
    boundary_points = np.zeros((num_boundary_points, DIM))
    for r in range(NUM_DIRECTIONS): #取0，1
        for s in range(NUM_DIRECTIONS): #取0，1
            boundary_points[r*NUM_DIRECTIONS + s, d] = shifted_q_point[d]
            boundary_points[r*NUM_DIRECTIONS + s, (d + 1) % DIM] = shifted_q_point[(d + 1) % DIM] + step / 2. * (2 * r - 1)
            boundary_points[r*NUM_DIRECTIONS + s, (d + 2) % DIM] = shifted_q_point[(d + 2) % DIM] + step / 2. * (2 * s - 1)
    #计算出每个shifted_q_point所对应的正方形面的4个顶点
    mapped_boundary_points = np.zeros((num_boundary_points, DIM))
    for i, b_point in enumerate(boundary_points):#把每个boundary_point映射到函数上（通过牛顿法来求解该mapped点的坐标）
        mapped_boundary_points[i] = sbm_map_newton(b_point, level_set, grad_level_set, ls_fn, ls_grad_fn)#用牛顿法把映射后对应的坐标表示出来  
    #下面一行表示将shifted_q_point映射到曲面函数上
    mapped_q_point = sbm_map_newton(shifted_q_point, level_set, grad_level_set, ls_fn, ls_grad_fn)
    #/为续行符
    weight = triangle_area(mapped_boundary_points[0], mapped_boundary_points[1], mapped_q_point) + \
             triangle_area(mapped_boundary_points[0], mapped_boundary_points[2], mapped_q_point) + \
             triangle_area(mapped_boundary_points[3], mapped_boundary_points[2], mapped_q_point) + \
             triangle_area(mapped_boundary_points[3], mapped_boundary_points[1], mapped_q_point)
    #weight值就是四个三角形面积之和
    return mapped_q_point, weight


def process_face(face, base, h, quad_level, ls_fn, ls_grad_fn):
    """对于每一个面，计算在曲面上对应的积分点和权重，quad_level是指该face被分割的程度
    """
    step = h / quad_level#把face的边长进行更近一步的划分
    mapped_quad_points = []#用这个列表存储映射后在曲面上的坐标点
    weights = []#用这个列表来储存权重，即四个三角形区域的面积
    element_id, face_number = face
    id_xyz = to_id_xyz(element_id, base)
    d = face_number // NUM_DIRECTIONS #NUM_DIRECTION = 2，还原参数d, 可以取0，1，2
    r = face_number % NUM_DIRECTIONS #还原参数r，可以取0，1
    shifted_quad_points = np.zeros((np.power(quad_level, 2), DIM)) #返回一个用0填充的数组
    for i in range(quad_level):
        for j in range(quad_level):
            shifted_quad_points[i * quad_level + j, d] = -DOMAIN_SIZE + (id_xyz[d] + r) * h
            shifted_quad_points[i * quad_level + j, (d + 1) % DIM] = -DOMAIN_SIZE + id_xyz[(d + 1) % DIM]* h + step / 2. + i * step
            shifted_quad_points[i * quad_level + j, (d + 2) % DIM] = -DOMAIN_SIZE + id_xyz[(d + 2) % DIM]* h + step / 2. + j * step
    #用上面这个循环算出了该面上的积分点的坐标值
    for shifted_q_point in shifted_quad_points:#对每一个shifted_quad_points作处理，计算对应的权重面积值
        mapped_quad_point, weight = estimate_weights(shifted_q_point, d, step, ls_fn, ls_grad_fn)
        mapped_quad_points.append(mapped_quad_point)
        weights.append(weight)
    return mapped_quad_points, weights


def generate_cut_elements(ls_fn):
    """第一个大的步骤，筛选出那些被曲面穿过的单元
    网格可以粗可以细，有不同的refinement level，对于每一个level，都筛选出响应的单元
    """
    start_refinement_level = 5
    end_refinement_level = 7
    start_base = np.power(DIVISION, start_refinement_level) #求几的几次方
    ids_cut = brute_force(start_base, ls_fn)#用“蛮力”遍历网格中的每一个单元，判断它是否被曲面零等高线穿过，将穿过的单元存到ids_cut
    total_ids = []
    total_refinement_levels = []#方括号是列表，list用于存储可改变的元素
    total_ids.append(ids_cut)
    total_refinement_levels.append(start_refinement_level)
    print(f"refinement_level {start_refinement_level}, length of inds {len(ids_cut)}") #打印一个字典

    # Note(Xue): 思考为什么这样做能够更加高效
    for refinement_level in range(start_refinement_level, end_refinement_level):
        ids_cut_new = []
        base = np.power(DIVISION, refinement_level)
        h = 2 * DOMAIN_SIZE / base
        for element_id in ids_cut:
            sub_ids = get_element_sub_ids(element_id, base)
            for sub_id in sub_ids:
                sub_id_x, sub_id_y, sub_id_z = to_id_xyz(sub_id, base * DIVISION)
                cut_flag, _, _ = is_cut(get_vertices(sub_id_x, sub_id_y, sub_id_z, h / DIVISION), ls_fn)
                if cut_flag:
                    ids_cut_new.append(sub_id)
        ids_cut = ids_cut_new
        total_ids.append(ids_cut)
        total_refinement_levels.append(refinement_level + 1)
        print(f"refinement_level {refinement_level + 1}, length of inds {len(ids_cut)}")

    total_ids = np.array(total_ids, dtype=object)
    total_refinement_levels = np.array(total_refinement_levels)
    print(f"len of total_refinement_levels {len(total_refinement_levels)}")
    return total_ids, total_refinement_levels#total_ids是一个二维列表


def compute_qw(total_ids, total_refinement_levels, ls_fn, ls_grad_fn, quad_level, mesh_index):
    """第二个大的步骤，计算积分点(quadrature point)以及积分点对应的权重(weight)
    有了积分点和权重，算曲面积分就易如反掌了。
    """
    ids_cut = total_ids[mesh_index]#total_ids是一个二维列表，ids_cut是一个一维列表，存有某一个refinement_level下的筛选后的编号
    refinement_level = total_refinement_levels[mesh_index]
    base = np.power(DIVISION, refinement_level)
    h = 2 * DOMAIN_SIZE / base
    print("\nrefinement_level is {} with h being {}, number of elements cut is {}".format(refinement_level, h, len(ids_cut)))
    faces = []
    for ele in range(len(ids_cut)):
        element_id = ids_cut[ele]#这个for循环是为了遍历每一个ids_cut中的每一个编号
        faces += neighbors(element_id, base, h, ls_fn)#找到surrogate boundary所用到的面（格子的id, 面对应的号码）

    mapped_quad_points = []
    weights = []
    for i, f in enumerate(faces):#遍历每一个面
        mapped_quad_points_f, weights_f = process_face(faces[i], base, h, quad_level, ls_fn, ls_grad_fn)
        mapped_quad_points += mapped_quad_points_f
        weights += weights_f
        if i % 100 == 0:
            print(f"Progress {(i + 1)/len(faces)*100:.5f}%, weights {np.sum(np.array(weights)):.5f}")

    mapped_quad_points = np.array(mapped_quad_points)
    weights = np.array(weights)

    return mapped_quad_points, weights#返回每一个积分点及其对应的权重
