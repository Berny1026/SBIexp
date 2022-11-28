import sys
import numpy as np
import matplotlib as plt
import copy

DIM = 3
DOMAIN_SIZE = 2 # 实际上我们定义的区域是[-2, 2]x[-2, 2]的正方形（2D情况）
DIVISION = 2
ORDER = 1
NUM_DIRECTIONS = 2

def level_set(point, ls_fn):#判断
    x = point[0]
    y = point[1]
    return ls_fn(x, y)

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
            for k in range(vertices_per_direction):
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

























