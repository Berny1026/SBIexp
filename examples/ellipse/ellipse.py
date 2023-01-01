import sys
import numpy as np
import matplotlib.pyplot as plt
import copy
import os

from sbi.Code2D import generate_cut_elements, compute_qw


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)

data_dir = os.path.join(os.path.dirname(__file__), 'data') 
numpy_dir = os.path.join(data_dir, f'numpy')
os.makedirs(numpy_dir, exist_ok=True)


def ellipse(x, y):
    """定义一个椭圆
    """
    value = np.power(x, 2)/16. + np.power(y, 2)/4. - 1
    return value


def grad_ellipse(x, y):
    """求圆上点的梯度
    """
    grad_x = 0.25*x
    grad_y = 0.5*y
    grad = np.array([grad_x, grad_y])
    return grad


def step1():
    total_ids, total_refinement_levels = generate_cut_elements(ellipse)
    np.savez(os.path.join(numpy_dir, f'cut_element_ids.npz'), ids=total_ids, 
        refinement_level=total_refinement_levels, allow_pickle=True)


def step2():
    data = np.load(os.path.join(numpy_dir, f'cut_element_ids.npz'), allow_pickle=True)
    total_ids = data['ids']
    total_refinement_levels = data['refinement_level']
    mesh_index = 1
    quad_level = 1
    mapped_quad_points, weights = compute_qw(total_ids, total_refinement_levels, ellipse, grad_ellipse, quad_level, mesh_index)
    np.savez(os.path.join(numpy_dir,  f'mesh_index_{mesh_index}_quad_level_{quad_level}_quads_and_weights.npz'),
        mapped_quad_points=mapped_quad_points, weights=weights, allow_pickle=True)

    # 我们可以手动计算出椭圆的周长，和SBI方法进行比较
    ground_truth = 4*np.pi + 8
    print(f"椭圆的周长 is {ground_truth:.5f}, SBI mehtod gives: {np.sum(weights):.5f}")


if __name__ == '__main__':
    step1()
    step2()

