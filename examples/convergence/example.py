import sys
import numpy as np
import matplotlib.pyplot as plt
import copy


np.set_printoptions(threshold=sys.maxsize, linewidth=1000, suppress=True)
np.set_printoptions(precision=5)
np.random.seed(1)


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
            for quad_level in quad_levels:
                compute_qw(quad_level, mesh_index, name_tests)

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
    convergence_tests()
    # plt.show()
