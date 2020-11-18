from matplotlib import transforms
from main_bayesian_exploration import *
from sklearn.decomposition import PCA
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from skimage.transform import resize

RESULTS_FOLDER = '../results/'

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.serif'] = 'Myriad Pro Regular'

label_dict = {
    "15mil": "$15mm$ inclusion",
    "10mil": "$10mm$ inclusion",
    "8mil": "$8mm$ inclusion",
    "5mil": "$5mm$ inclusion",
    "na": "No inclusion"
}

label_dict_abbrv = {
    "15mil": "$15mm$ incl.",
    "10mil": "$10mm$ incl.",
    "8mil": "$8mm$ incl.",
    "5mil": "$5mm$ incl.",
    "na": "No incl."
}

color_label_dict = {
    "15mil": "#ff0000",
    "10mil": "black",
    "8mil": "blue",
    "5mil": "#fcb900",
    "na": "#2ec400"
}

exp_dictionary = {
    'exploration_training_phantom_1': "Training Phantom 1",
    'exploration_training_phantom_2': "Training Phantom 2",
    'abdominal_phantom': "Abdominal Phantom",
    'complexity_demonstration': "ON-OFF Test",
}


def find_action(action, actions):
    idx, A, B = action
    for act_id, alphas, etas in actions:
        if alphas == A and etas == B:
            return act_id
    return None


def histogram_plot(hist_array, xlabel='$principal components$', ylabel='$explained variance (%)$', xtick_prefix='p'):
    hist_fig = plt.figure(figsize=(13, 8))
    ax = hist_fig.add_subplot(111)

    # necessary variables
    ind = np.arange(1, len(hist_array) + 1)  # the x locations for the groups
    width = 0.85

    # the bars
    ax.bar(ind, hist_array * 100, width, color='black')

    xTickMarks = ['$' + xtick_prefix + '_{' + str(i) + '}$' for i in ind]
    ax.set_xticks(ind)
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=22)
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=34)

    return hist_fig


def generate_pca_histogram(actions=None, original_belief_data=None, folder="", rank=0, show=False, save=True):
    plt.close('all')

    pss = []
    for action in actions:
        obj_keys = sorted(original_belief_data[action[0]].keys())
        dataset = np.zeros((1, original_belief_data[action[0]][obj_keys[0]].shape[1]))
        for obj_key in obj_keys:
            dataset = np.append(dataset, original_belief_data[action[0]][obj_key], axis=0)
        pca = PCA(n_components=min(dataset[1:, :].shape))
        pca.fit(dataset[1:, :])
        ps = np.sort(np.var(pca.transform(dataset[1:, :]), axis=0) / \
                     np.sum(np.var(dataset[1:, :], axis=0)))
        ps = np.concatenate((ps[::-1], np.zeros(dataset[1:, :].shape[0] - len(ps))), axis=0)[:20]
        pss += [ps]
    ps_data = np.array(pss)
    meanP = np.average(ps_data, axis=0)
    errP = np.std(ps_data, axis=0)  # / np.sqrt(ps_data.shape[0])
    hist_fig = plt.figure(figsize=(13, 8))
    ax = hist_fig.add_subplot(111)
    ax.set_xlabel("$principal\ components$", fontsize=28)
    ax.set_ylabel("$explained\ variance (\%)$", fontsize=34)
    ax.set_ylim([0, 100])
    # necessary variables
    ind = np.arange(1, len(meanP) + 1)  # the x locations for the groups
    xTickMarks = ['$p_{' + str(i) + '}$' for i in ind]
    width = 0.85
    ax.bar(ind, meanP * 100, width, color='black', yerr=errP * 100,
           error_kw=dict(ecolor='red', lw=3, capsize=3, capthick=2))
    plt.xticks(ind, xTickMarks, fontsize=22)
    plt.yticks(fontsize=22)
    hist_fig.tight_layout()

    if save:
        filename = '{}{}-pca.png'.format(folder, rank)
        hist_fig.savefig(filename, bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    return hist_fig


def generate_motion_profile(data=None, action=None, legend=False, show=False, save=False,
                            plot_accuracy=False, figure=None, grdspc=None, label_pos=None, no_title=False):
    pdfs = data["pdfs"]
    idx = [i for i, act in enumerate(sorted(pdfs.keys())) if act == action[0]][0]
    accuracy = data["action_accuracies"][idx][1]
    action_benefits = data["unbiased_action_benefits"][idx]
    rank = np.where(np.array(data["unbiased_ranked_actions"]) == action[0])[0][0]

    if figure is not None:
        fig = figure
        ax = figure.add_subplot(grdspc, projection='3d')
    else:
        plt.close('all')
        fig = plt.figure(1, figsize=(13, 10))
        ax = fig.gca(projection='3d')

    if not no_title:
        if plot_accuracy:
                ax.set_title('$\\bf{Rank = ' + str(rank) + '}'+'$\nAccuracy = {:.2f}\n$B_m^u$ = {:.2f}\n'.format(
                    accuracy, action_benefits), loc="center", fontsize=42)
        else:
                ax.set_title('$\\bf{Rank = ' + str(rank) + '}'+'$\n$B_m^u$ = {:.2f}\n'.format(
                    action_benefits), loc="center", fontsize=42)

    ax.set_xlabel('$Rx\ (deg)$', fontsize=34, labelpad=30)
    ax.set_ylabel('$Ry\ (deg)$', fontsize=34, labelpad=30)
    ax.set_zlabel('$Z\ (mm)$', fontsize=34, labelpad=30)
    ax.xaxis.set_tick_params(labelsize=30)
    ax.yaxis.set_tick_params(labelsize=30)
    ax.zaxis.set_tick_params(labelsize=30)
    ax.set_xlim([-7, 7])
    ax.set_ylim([-7, 7])
    ax.view_init(azim=-45)

    etas = np.array(action[1])
    alphas = np.array(action[2])

    increments = 0.01
    maximum_time = 3.  # 3 seconds
    current_time = 0.00

    positions = []

    max_iterations = np.floor((maximum_time - current_time) / increments)
    current_iteration = 0

    middle_legend_done = False

    robot_position = np.array([0., 0., 0.])
    robot_rotation = np.array([0., 0., 0.])

    zs = []
    rxs = []
    rys = []

    initial_position = alphas * np.cos(etas * current_time)

    while current_time <= maximum_time:

        current_time += increments

        if 35 > current_iteration:
            color = 'green'
            alpha = .7
        elif 4 > max_iterations - current_iteration:
            color = 'red'
            alpha = .7
        else:
            color = 'grey'
            alpha = .25

        if current_iteration == 0:
            legend_text = "initial end-effector pose"
        elif current_iteration == max_iterations - 1:
            legend_text = "final end-effector pose"
        elif middle_legend_done is False and color == 'grey':
            legend_text = "end-effector pose"
            middle_legend_done = True
        else:
            legend_text = False

        velocities = -alphas * etas * np.sin(etas * current_time)
        velocities = np.array([0 if math.isnan(elem) or math.isinf(elem) else elem for elem in
                               velocities])
        current_position = alphas * np.cos(etas * current_time)
        robot_position += velocities[:3] * current_time * 1000  # in mm
        robot_rotation += velocities[3:] * current_time  # in radiants

        positions += [robot_position]

        x, y, z = robot_position
        Rx, Ry, Rz = robot_rotation

        zs += [(current_position[2] - initial_position[2]) * 1000]
        rxs += [np.rad2deg(current_position[3] - initial_position[3])]
        rys += [np.rad2deg(current_position[4] - initial_position[4])]

        # x2 = x + np.sin(Ry)
        # y2 = y + -np.sin(Rx ) *np.cos(Ry)
        # z2 = z + np.cos(Rx ) *np.cos(Ry)

        if legend_text:
            ax.plot([0, rxs[-1]], [0, rys[-1]], [zs[-1] - initial_position[2], zs[-1] - initial_position[2] + 10],
                    color=color, alpha=alpha, label=legend_text, zorder=10)
        else:
            ax.plot([0, rxs[-1]], [0, rys[-1]], [zs[-1] - initial_position[2], zs[-1] - initial_position[2] + 10],
                    color=color, alpha=alpha, zorder=10)

        current_iteration += 1
    ax.scatter(rxs, rys, zs - initial_position[2] + 10, label="Rx, Ry, z robot control", zorder=0, color='blue')
    ax.view_init(azim=30)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(handle) for handle in handles]
        [handle.set_linewidth(10) for handle in handles]
        [handle.set_alpha(1) for handle in handles]
        if save:
                ax.legend(handles=handles, labels=labels, fontsize=48, loc='upper center', bbox_to_anchor=(0., -2.),
                            ncol=4, fancybox=True)
        else:
            if label_pos == 'left':
                ax.legend(handles=handles, labels=labels, fontsize=48, loc='upper right', bbox_to_anchor=(4.36, 0.),
                        ncol=4, fancybox=True)
            else:
                ax.legend(handles=handles, labels=labels, fontsize=48, loc='upper right', bbox_to_anchor=(4.86, 0.),
                            ncol=4, fancybox=True)

    if save:
        filename = '{}{}_{}-action_profile.png'.format(RESULTS_FOLDER, rank, action[0])
        # fig.savefig(filename, bbox_inches="tight", dpi=300)
        fig.savefig(filename, dpi=300)

    if show:
        plt.show()

    return fig


def generate_control_profile(data=None, action=None, rank=None, folder='', legend=False, show=False, save=False):
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    ax.set_xlabel('$time\ (ms)$', fontsize=28)
    ax.set_ylabel('$displacement\ (mm)$', fontsize=28)
    ax.yaxis.set_tick_params(labelsize=22)
    ax.xaxis.set_tick_params(labelsize=22)

    etas = np.array(action[1])
    alphas = np.array(action[2])

    increments = 0.01
    maximum_time = 6.  # 3 seconds
    current_time = 0.00

    positions = []

    max_iterations = np.floor((maximum_time - current_time) / increments)
    current_iteration = 0

    middle_legend_done = False

    robot_position = np.array([0., 0., 0.])
    robot_rotation = np.array([0., 0., 0.])

    zs = []
    rxs = []
    rys = []

    initial_position = alphas * np.cos(etas * current_time)

    while current_time <= maximum_time:

        current_time += increments

        if 35 > current_iteration:
            color = 'green'
        elif 4 > max_iterations - current_iteration:
            color = 'red'
        else:
            color = 'grey'

        if current_iteration == 0:
            legend_text = "initial end-effector pose"
        elif current_iteration == max_iterations - 1:
            legend_text = "final  end-effector pose"
        elif middle_legend_done is False and color == 'grey':
            legend_text = "palpation's end-effector pose"
            middle_legend_done = True
        else:
            legend_text = False

        velocities = -alphas * etas * np.sin(etas * current_time)
        velocities = np.array([0 if math.isnan(elem) or math.isinf(elem) else elem for elem in
                               velocities])
        current_position = alphas * np.cos(etas * current_time)
        robot_position += velocities[:3] * current_time * 1000  # in mm
        robot_rotation += velocities[3:] * current_time  # in radiants

        positions += [robot_position]

        zs += [(current_position[2] - initial_position[2]) * 1000]
        rxs += [np.rad2deg(current_position[3] - initial_position[3])]
        rys += [np.rad2deg(current_position[4] - initial_position[4])]


        current_iteration += 1

    plt.plot(np.array((range(len(rxs))))*10, rxs, label="Rx", lw=3)
    plt.plot(np.array((range(len(rys))))*10, rys, label='Ry', lw=3)
    plt.plot(np.array((range(len(zs))))*10, zs, label='Z', lw=3)

    ax.legend(fontsize=28, loc='upper right', bbox_to_anchor=(.9, 1.22), ncol=4, fancybox=True)

    if save:
        filename = '{}{}_{}-control_action_profile.png'.format(RESULTS_FOLDER, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig


def generate_belief_state(data=None, action=None, legend=False, folder="", show=False, save=False,
                            figure=None, grdspc=None):
    pdfs = data['pdfs']
    rank = np.where(np.array(data["unbiased_ranked_actions"]) == action[0])[0][0]
    reduced_data = data['reduced_belief_data']
    # reduced_test_data = data['reduced_test_data']
    colors = ['#025df0', '#ff7700', '#08c453', '#850f67', '#c4c4cc', '#000000']

    act = action[0]

    if figure is not None:
        fig = figure
        grid = gridspec.GridSpecFromSubplotSpec(5, 5, subplot_spec=grdspc)
    else:
        fig = plt.figure(figsize=(13, 10), constrained_layout=True)
        grid = gridspec.GridSpec(ncols=5, nrows=5, figure=fig)
    ax_gauss = fig.add_subplot(grid[:-1, :])
    ax_1d = fig.add_subplot(grid[-1, :], sharex=ax_gauss)
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    ax_1d.set_xlabel('$\\vec{p}_1$', fontsize=48)
    ax_gauss.set_ylabel('$p(x)$', fontsize=48)
    plt.autoscale(True)

    ax_gauss.yaxis.set_tick_params(labelsize=30)
    ax_1d.xaxis.set_tick_params(labelsize=30)
    ax_1d.yaxis.set_ticks([])
    ax_gauss.label_outer()
    ax_1d.label_outer()

    global_min = None
    global_max = None

    for obj in pdfs[act].keys():
        mu, sig = pdfs[act][obj]
        xmin = mu - 4 * sig
        xmax = mu + 4 * sig

        if global_min is None or global_min > xmin:
            global_min = xmin
        if global_max is None or global_max < xmax:
            global_max = xmax

    if global_min == global_max:
        xs = [0] * 100
    else:
        xs = np.arange(global_min, global_max, (global_max - global_min) / 100)

    for i, obj in enumerate(pdfs[act].keys()):
        mu, sig = pdfs[act][obj]
        ax_1d.plot(
            reduced_data[act][obj][:, 0], [0] * len(reduced_data[act][obj][:, 0]),
            marker='|',
            linestyle='None',
            markersize=35,
            markerfacecolor=color_label_dict[obj],
            markeredgewidth=7,
            markeredgecolor=color_label_dict[obj],
            label="{}".format(label_dict[obj]),
            alpha=0.8
        )

        ax_gauss.plot(xs, gaussian(xs, mu, sig), label=label_dict[obj], c=color_label_dict[obj], linewidth=6)

    # if legend:
    handles1, labels1 = ax_gauss.get_legend_handles_labels()
    handles2, labels2 = ax_1d.get_legend_handles_labels()
    handles = [copy.copy(handle) for handle in handles1 + handles2]
    [handle.set_linewidth(8) for handle in handles]
    [handle.set_alpha(1) for handle in handles]

    if save:
        filename = '{}{}_{}-belief_state.png'.format(RESULTS_FOLDER, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return handles, labels1 + labels2


def generate_figure_3(data, save_key='', show=False, save=True):

    # plt.close('all')
    actions = data['actions']
    ranked_actions = data['unbiased_ranked_actions']
    folder = data['results_folder']

    fig = plt.figure(figsize=(65, 26), constrained_layout=True)
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    grid = gridspec.GridSpec(ncols=4, nrows=9, figure=fig)  #, wspace=.8, hspace=.1
    grid.update(wspace=0.25, hspace=0.35)  # set the spacing between axes.

    y_coord_labels = ['$i)$', '$ii)$']
    label_grids = [grid[0:4, :], grid[5:, :]]
    for i in range(2):
        ax_main = fig.add_subplot(label_grids[i], frameon=False)
        ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
        ax_main.set_ylabel(y_coord_labels[i], fontsize=90, fontweight='bold', labelpad=140, rotation=0)

    handles, labels = (None, None)
    ranked_idxs_to_extract = list(np.ceil(np.arange(0, len(actions), len(actions)/3)).astype(np.int))+[len(actions)-1]
    for i, idx in enumerate(ranked_idxs_to_extract):
        generate_motion_profile(action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                                data=data,
                                show=show,
                                legend=idx == 0,
                                save=False,
                                figure=fig,
                                grdspc=grid[0:4, i],
                                label_pos='left')

        print("generated subfig {}-{} {}, out of {} done".format(1, i, i*3+2, 12))
        handles, labels = generate_belief_state(data=data,
                              action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                              folder=folder,
                              show=show,
                              legend=idx == 0,
                              save=False,
                              figure=fig,
                              grdspc=grid[5:, i])
        print("generated subfig {}-{} {}, out of {} done".format(2, i, i*3+3, 12))

    if len(data['pdfs'][list(data['pdfs'].keys())[0]].keys()) == 4:
        ax.legend(handles=handles, labels=labels, fontsize=48, loc='lower center',
              bbox_to_anchor=(.5, -0.17), ncol=4, fancybox=True)
    else:
        ax.legend(handles=handles, labels=labels, fontsize=48, loc='lower center',
              bbox_to_anchor=(.48, -0.17), ncol=6, fancybox=True)

    if save:
        filename = '{}{}Figure3.png'.format(RESULTS_FOLDER, save_key)
        fig.savefig(filename, bbox_inches="tight", dpi=130)
        plt.close('all')
    return fig


def generate_figure_6(data, save_key='', show=False, save=True):

    plt.close('all')

    folder = data['results_folder']

    ranked_actions = data['unbiased_ranked_actions']
    actions = data['actions']
    fig = plt.figure(figsize=(55, 25), constrained_layout=True)
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    grid = gridspec.GridSpec(ncols=4, nrows=20, figure=fig)  # wspace=.8, hspace=.1
    grid.update(wspace=0.25, hspace=0.35)  # set the spacing between axes.

    y_coord_labels = ['i)', 'ii)', 'iii)']
    label_grids = [grid[0:7, :], grid[9:14, :], grid[15:, :]]
    for i in range(3):
        ax_main = fig.add_subplot(label_grids[i], frameon=False)
        ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
        ax_main.set_ylabel(y_coord_labels[i], fontsize=90, labelpad=140, rotation=0)

    handles, labels = (None, None)
    ranked_idxs_to_extract = list(np.ceil(np.arange(0, len(actions), len(actions)/3)).astype(np.int))+[len(actions)-1]
    # ranked_idxs_to_extract = [0, 1, -2, -1]
    for i, idx in enumerate(ranked_idxs_to_extract):
        generate_motion_profile(action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                                data=data,
                                show=show,
                                legend=idx == 0,
                                save=False,
                                plot_accuracy=True,
                                figure=fig,
                                grdspc=grid[0:7, i])
        print("generated subfig {}-{} {}, out of {} done".format(1, i, i*3+2, 12))
        generate_raw_data_figure(data=data,
                                 action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                                 rank=idx,
                                 folder=folder,
                                 show=show,
                                 colorbar=i == len(ranked_idxs_to_extract)-1,
                                 save=False,
                                 figure=fig,
                                 grdspc=grid[9:14, i])
        print("generated subfig {}-{} {}, out of {} done".format(0, i, i*3+1, 12))
        handles, labels = generate_belief_state(data=data,
                              action=[action for action in actions if action[0] == ranked_actions[idx]][0],
                              folder=folder,
                              show=show,
                              legend=idx == 0,
                              save=False,
                              figure=fig,
                              grdspc=grid[15:, i])
        print("generated subfig {}-{} {}, out of {} done".format(2, i, i*3+3, 12))

    if len(data['pdfs'][list(data['pdfs'].keys())[0]].keys()) == 4:
        ax.legend(handles=handles, labels=labels, fontsize=48, loc='lower center',
              bbox_to_anchor=(.5, -0.17), ncol=4, fancybox=True)
    else:
        ax.legend(handles=handles, labels=labels, fontsize=48, loc='lower center',
              bbox_to_anchor=(.48, -0.17), ncol=6, fancybox=True)

    if save:
        filename = '{}{}Figure6.png'.format(RESULTS_FOLDER, save_key)
        fig.savefig(filename, bbox_inches="tight", dpi=150)
        plt.close('all')
    return fig


def generate_figure_4(all_bayesian_data, all_systematic_data, show=False, save=True):

    plt.close('all')
    folder = all_bayesian_data[list(all_bayesian_data.keys())[0]]['results_folder']

    fig = plt.figure(figsize=(70, 20), constrained_layout=True)
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    grid = gridspec.GridSpec(nrows=1, ncols=3, figure=fig)  # wspace=.8, hspace=.1
    grid.update(wspace=0.3, hspace=0.5)  # set the spacing between axes.

    col_no = 0
    sub_axes = []
    main_xlim = None
    main_ylim = None
    handles, labels = (None, None)
    for exp_no in all_bayesian_data.keys():
        bayesian_data = all_bayesian_data[exp_no]

        ax_sub, handles, labels, xlim, ylim = generate_figure_ranked_acc(exp_number=exp_no,
                                                     data=bayesian_data,
                                                     legend=col_no == 0,
                                                     figure=fig,
                                                     grdspc=grid[:, col_no],
                                                     show=False,
                                                     save=False)
        sub_axes += [ax_sub]

        if main_xlim is None or main_ylim is None:
            main_xlim = xlim
            main_ylim = ylim
        else:
            main_xlim = [min(main_xlim[0], xlim[0]), max(main_xlim[1], xlim[1])]
            main_ylim = [min(main_ylim[0], ylim[0]), max(main_ylim[1], ylim[1])]

        print("generated subfig {}-{} {}, out of {} done".format(0, col_no, col_no*2+1, 6))
        col_no += 1

    for sub_ax in sub_axes:
        sub_ax.set_xlim(main_xlim)
        sub_ax.set_ylim(main_ylim)

    ax.legend(handles=handles, labels=labels, fontsize=75, loc='lower center',
              bbox_to_anchor=(.5, -0.75), ncol=2, fancybox=True)

    if save:
        filename = '{}Figure4.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=60)
        plt.close('all')
    return fig


def generate_figure_5(all_bayesian_data, all_systematic_data, show=False, save=True):
    plt.close('all')
    folder = all_bayesian_data[list(all_bayesian_data.keys())[0]]['results_folder']

    fig = plt.figure(figsize=(60, 20), constrained_layout=True)

    grid = gridspec.GridSpec(nrows=3, ncols=4, figure=fig)  # wspace=.8, hspace=.1
    grid.update(wspace=0.3, hspace=0.5)  # set the spacing between axes.

    y_coord_labels = ['i)', 'ii)', 'iii)']
    for i in range(3):
        ax_main = fig.add_subplot(grid[i, 1:], frameon=False)
        ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
        ax_main.set_ylabel(y_coord_labels[i], fontsize=90, labelpad=135, rotation=0)

    col_no = 1
    for exp_no in all_bayesian_data.keys():
        bayesian_data = all_bayesian_data[exp_no]
        systematic_data = all_systematic_data[exp_no]

        generate_figure_10(exp_number=exp_no,
                           data_bayesian=bayesian_data,
                           data_systematic=systematic_data,
                           legend=col_no == 1,
                           figure=fig,
                           grdspc=grid[0, col_no],
                           show=False,
                           save=False)
        print("generated subfig {}-{} {}, out of {} done".format(0, col_no, col_no*2+1, 6))

        generate_belief_change_figure(bayesian_data=bayesian_data,
                                      systematic_data=systematic_data,
                                      colorbar=col_no == len(all_bayesian_data.keys()),
                                      figure=fig,
                                      grdspc=grid[1:, col_no],
                                      show=False,
                                      save=False)
        print("generated subfig {}-{} {}, out of {} done".format(1, col_no, col_no*2+2, 6))
        col_no += 1

    if save:
        filename = '{}Figure5.png'.format(RESULTS_FOLDER)
        # Save just the portion _inside_ the second axis's boundaries
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        points = bbox.get_points()
        new_points = points*np.array([[1., 1.], [1.02, 1.05]])  # 75% of the width instead of whole fig
        fig.savefig(filename, bbox_inches=transforms.Bbox(new_points), dpi=200)
    return fig


def generate_figure_ranked_acc(data=None, exp_number=0, legend=False, figure=None, grdspc=None, show=False, save=True):
    ranked_actions = data['unbiased_ranked_actions']
    action_benefits = data["unbiased_action_benefits"]
    action_accuracies = data['action_accuracies']
    # pdfs = data['pdfs']
    folder = data['results_folder']

    action_benefits = sorted(action_benefits, reverse=True)

    if figure is not None:
        fig = figure
        # grid = gridspec.GridSpecFromSubplotSpec(100, 100, subplot_spec=grdspc)
        ax = fig.add_subplot(grdspc)
    else:
        fig = plt.figure(1, figsize=(13, 10))
        ax = fig.add_subplot(111)

    if exp_number is not None:
       ax.set_title("{}\n".format(exp_dictionary[exp_number]), loc="center", fontsize=100, fontweight='bold')
    ax.yaxis.set_tick_params(labelsize=42)
    ax.xaxis.set_tick_params(labelsize=42)


    ranked_accuracies = []
    for act in ranked_actions:
        ranked_accuracies += [int(accuracy[1]*100) for accuracy in action_accuracies if accuracy[0] == act]

    plt.plot(sorted(action_benefits, reverse=True), ranked_accuracies,
             linewidth=9,
             linestyle='-',
             color='k',
             marker='*',
             markersize=12,
             # alpha=.7
             )

    plt.plot(sorted(action_benefits, reverse=True)[0], ranked_accuracies[0],
             linestyle=None,
             marker='o',
             markersize=62,
             markerfacecolor='green',
             markeredgewidth=4,
             markeredgecolor='k',
             alpha=.9,
             label="lowest ranked action"
             )
    plt.plot(sorted(action_benefits, reverse=True)[-1], ranked_accuracies[-1],
             linestyle=None,
             marker='o',
             markersize=72,
             markerfacecolor='red',
             markeredgewidth=10,
             markeredgecolor='k',
             alpha=.9,
             label="highest ranked action"
             )

    ax.set_xlabel('$Unbiased\ Benefit\ Estimator\ (B^u)$', fontsize=75)
    ax.set_ylabel('$Classification\ Accuracy (\%)$', fontsize=75)
    ylim = ax.get_ylim()
    xlim = ax.get_xlim()


    handles, labels = ax.get_legend_handles_labels()
    handles = [copy.copy(handle) for handle in handles]
    [handle.set_markersize(90) for handle in handles]
    [handle.set_alpha(1) for handle in handles]

    if save:
        filename = '{}fig6.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return ax, handles, labels, xlim, ylim


def generate_motion_param_comparison_figure1(data, show=False, save=False):
    plt.close('all')
    actions = data['actions']
    ranked_actions = data['unbiased_ranked_actions']
    folder = data['results_folder']

    fig = plt.figure(figsize=(13, 10))
    grid = gridspec.GridSpec(ncols=1, nrows=6, figure=fig, wspace=.5, hspace=.5)

    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('Ranked Motions', fontsize=28, labelpad=20)
    plt.ylabel('$Action\ Parameters$', fontsize=28, labelpad=60)

    params = [(0, (1, 2), "$\omega^Z$"),
              (1, (1, 3), "$\omega^{Rx}$"),
              (2, (1, 4), "$\omega^{Ry}$"),
              (3, (2, 2), "$A^{Z}$"),
              (4, (2, 3), "$A^{Rx}$"),
              (5, (2, 4), "$A^{Ry}$")]

    for i, idxes, param in params:
        if i == 0:
            ax_subplot = fig.add_subplot(grid[i, :])
        else:
            ax_subplot = fig.add_subplot(grid[i, :], sharex=ax_subplot)

        # ax_1d.set_xlabel('$\\vec{p}_1$', fontsize=48)
        ax_subplot.set_ylabel(param, fontsize=22)

        param_value = []
        for ranked_action in ranked_actions:
            act = [action for action in actions if action[0]==ranked_action][0]
            param_value += [act[idxes[0]][idxes[1]]]
        ax_subplot.plot(param_value)
        ax_subplot.get_xaxis().set_ticks(range(0, 64, 3))
        if i == params[-1][0]:
            ax_subplot.xaxis.set_tick_params(labelsize=18)
        ax_subplot.yaxis.set_tick_params(labelsize=14)

    if save:
        filename = '{}-action_comparison.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return None


def generate_belief_change_figure(bayesian_data=None, systematic_data=None, colorbar=False, figure=None,
                                  grdspc=None, show=False, save=False):
    plt.close('all')
    folder = bayesian_data['results_folder']

    if figure is not None:
        fig = figure
        grid = gridspec.GridSpecFromSubplotSpec(10, 10, subplot_spec=grdspc, wspace=0.05, hspace=0.05)
    else:
        fig = plt.figure(0, figsize=(15, 10))
        grid = gridspec.GridSpec(nrows=50, ncols=2, figure=fig)

    bar_plot_grid = gridspec.GridSpecFromSubplotSpec(10, 1, subplot_spec=grdspc)
    total_ax = fig.add_subplot(bar_plot_grid[1:-1, :], frameon=False)
    total_ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    bayesian_benefits = np.array(bayesian_data['iteration_benefits']).T
    systematic_data_benefits = np.array(systematic_data['iteration_benefits']).T

    max_x = max(bayesian_benefits.shape[1], systematic_data_benefits.shape[1])

    for j, iterations_benefits in enumerate([bayesian_benefits, systematic_data_benefits]):
        if colorbar:
            ax = fig.add_subplot(
                grid[j*5+2:j*5+4, :-1]
            )
            # ax_bar = fig.add_subplot(
            #     grid[j*5+1:j*5+5, -1]
            # )
        else:
            ax = fig.add_subplot(
                grid[j*5+2:j*5+4, 1:]
            )

        iteration_ranks = np.zeros_like(iterations_benefits)
        for i in range(iterations_benefits.shape[1]):
            # attribute a rank to every action
            iteration_ranks[iterations_benefits[:, i].argsort(), i] = list(range(iteration_ranks.shape[0]))
        # order the rank by the final ordering
        iteration_ranks = iteration_ranks[iterations_benefits[:, -1].argsort(), :]

        if iteration_ranks.shape[1] < max_x:
            # padding for missing data
            iteration_ranks = np.append(
                arr=iteration_ranks,
                values=np.broadcast_to(iteration_ranks[:, -1].reshape(-1, 1),
                                       (iteration_ranks.shape[0],
                                        max_x-iteration_ranks.shape[1])), axis=1)

        # b_coeffs = np.array(ordered_iteration_benefits).T
        image_to_print = resize(iteration_ranks,
                                (iteration_ranks.shape[0]*2.5, iteration_ranks.shape[1]),
                                anti_aliasing=True)
        im = ax.imshow(iteration_ranks, cmap='gist_gray', aspect='auto')#, vmin=0, vmax=64)
                       # norm=colors.LogNorm(vmin=np.min(iteration_ranks),
                       #                     vmax=np.max(iteration_ranks)))

        ax.xaxis.set_tick_params(labelsize=24)
        ax.yaxis.set_tick_params(labelsize=24)
        ax.set_xlabel('$Iteration\ number$', fontsize=34)
        ax.set_ylabel('$Ranked$ \n$Action\ Identifiers$', fontsize=34)

        # add converted iterations in hours

        ax2 = fig.add_axes(ax.get_position(), frameon=False)
        ax2.tick_params(labelbottom=False, labeltop=True, labelleft=False, labelright=False,
                        bottom=False, left=False, right=False)

        ax1Xs = ax.get_xticks()
        upperXTickMarks = ["{:.2f}".format(x*.0302+.20) for x in ax1Xs]
        for t, tick in enumerate(upperXTickMarks):
            upperXTickMarks[t] = "{}:{}h".format(tick.split('.')[0], str(int(float(tick.split('.')[1])*60/100)))
        ax2.set_xticks(ax1Xs)
        ax2.set_xticklabels(upperXTickMarks, minor=False)
        ax2.set_xbound(ax.get_xbound())
        ax2.text(0.5, 1.28, "$Palpation\ exploration\ time$",
                 horizontalalignment='center',
                 fontsize=28,
                 transform=ax2.transAxes)
        ax2.xaxis.set_tick_params(labelsize=24)

    if colorbar:
        divider = make_axes_locatable(total_ax)
        cax = divider.append_axes("right", size="8%", pad="20%")
        cbar = plt.colorbar(im, cax=cax)#, ticks=np.arange(0.0, 64., 1.))
        cbar.ax.get_yaxis().labelpad = 15
        cbar.ax.tick_params(labelsize=38, rotation=90)
        cbar.ax.set_ylabel('$B_m^u\ Rank$', rotation=270, fontsize=38, labelpad=90)

    if save:
        filename = '{}-iteration_benefits.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=200)

    if show:
        plt.show()

    return True


def generate_rank_change_figure(data, show=False, save=False):
    plt.close('all')
    iteration_rankings = data['iteration_rankings']
    folder = data['results_folder']

    ordered_itaration_rankings = []
    for ranking in iteration_rankings:
        ordered_ranking = []
        for act in iteration_rankings[-1]:
            ordered_ranking += [np.where(np.array(ranking) == act)[0][0]]
        ordered_itaration_rankings += [ordered_ranking]

    fig = plt.figure(0, figsize=(13, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.xlabel('$Iteration\ number$', fontsize=22)
    plt.ylabel('$Action\ Itentifiers$', fontsize=22)

    b_coeffs = np.array(ordered_itaration_rankings).T

    im = ax.imshow(b_coeffs, cmap='gist_gray')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.15)
    cbar = plt.colorbar(im, cax=cax)
    cbar.ax.get_yaxis().labelpad = 15
    cbar.ax.set_ylabel('$b^u_m$', rotation=270)

    if save:
        filename = '{}-iteration_rankings.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return True

def time_complexity_figure(data, show=False, save=False):
    plt.close('all')
    fig = plt.figure(0, figsize=(13, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.xlabel('$Palpation Time$', fontsize=22)
    plt.ylabel('$Accuracy$', fontsize=22)

    for phantom in data[2].keys():
        complexity_error_accs = []
        complexity_avg_accs = []
        complexity_max_accs = []
        for time_limit in range(2, 10):
            if phantom >= 7 or (phantom < 7 and time_limit <= 6):
                action_accuracies = data[time_limit][phantom]['action_accuracies']
                accs = [acc for act, acc in action_accuracies]
                complexity_error_accs += [np.std(accs) / np.sqrt(len(accs))]
                complexity_avg_accs += [np.average(accs)]
                complexity_max_accs += [np.max(accs)]
        # plt.errorbar(list(range(1, len(complexity_avg_accs) + 1)), complexity_avg_accs, yerr=complexity_error_accs,
        #              label="{}".format(exp_dictionary[phantom]))
        plt.errorbar(list(range(1, len(complexity_avg_accs) + 1)), complexity_max_accs, yerr=complexity_error_accs,
                     label="{} max accuracy".format(exp_dictionary[phantom]))
    xTickMarks = ["{:.1f}".format(time_tick / 2 + 1.) for time_tick in range(1, len(complexity_avg_accs) + 1)]
    ax.set_xticks(range(1, len(complexity_avg_accs) + 1))
    xtickNames = ax.set_xticklabels(xTickMarks)
    plt.setp(xtickNames, fontsize=22)
    plt.legend()

    if save:
        filename = '{}-complexity.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()


def axis_complexity_figure(data, show=False, save=False):
    plt.close('all')
    fig = plt.figure(0, figsize=(10, 8))
    ax = fig.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=18)
    ax.xaxis.set_tick_params(labelsize=18)
    plt.xlabel('$Number$ $of$ $non$-$zero$ $control$ $parameters$', fontsize=24)
    plt.ylabel('$Accuracy$', fontsize=26)
    actions = data['actions']
    action_accuracies = data['action_accuracies']

    param_complexity = {0: [], 1: [], 2: [], 3: [], 4: [], 5: [], 6: []}

    for act_id, acc in action_accuracies:
        alphas, etas = [(alphas, etas) for idx, alphas, etas in actions if idx == act_id][0]

        cnt = 0
        for prm in alphas + etas:
            if prm != 0:
                cnt += 1

        param_complexity[cnt] += [acc]

    xs = sorted(param_complexity.keys())
    ys_avg = [np.average(param_complexity[key]) for key in xs]
    ys_err = [np.sqrt(np.std(param_complexity[key]))/len(param_complexity[key]) for key in xs]
    plt.errorbar(xs[1:], ys_avg[1:],
                 yerr=ys_err[1:],
                 elinewidth=5,
                 ecolor='k',
                 lw=4,
                 c='g',
                 solid_capstyle='projecting',
                 capsize=15,
                 label="Average diagnostic accuracy")
    ax.grid(alpha=0.5, linestyle=':')
    ax.legend(fontsize=22)

    if save:
        filename = '{}-param_complexity.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()


def sample_accuracy_figure(data, show=False, save=False):
    plt.close('all')
    fig = plt.figure(0, figsize=(13, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.xlabel('$Number\ of\ training\ samples$', fontsize=22)
    plt.ylabel('$Maximum\ accuracy$', fontsize=22)

    if 'results_folder' in data.keys():
        folder = data['results_folder']
        actions = data['actions']
        running_accuracies = data['running_accuracies']
        xs = [sample for sample, _ in running_accuracies[list(running_accuracies.keys())[0]]]

        ys_avg = []
        ys_err = []
        for sample in xs:
            accs = []
            for act_id, _, _ in actions:
                accs += [acc for smp, acc in running_accuracies[act_id] if smp==sample]
            ys_avg += [np.max(accs)]
            ys_err += [np.sqrt(np.std(accs))/len(accs)]

        plt.errorbar(xs, ys_avg, yerr=ys_err, uplims=True, lolims=True, elinewidth=3, lw=3, c='k')
    else:
        colors = ['#000dff', '#26d100', '#a31000', '#00a2ad']
        for i, key in enumerate(data.keys()):
            actions = data[key]['actions']
            running_accuracies = data[key]['running_accuracies']
            xs = [sample for sample, _ in running_accuracies[list(running_accuracies.keys())[0]]]

            ys_avg = []
            ys_err = []
            for sample in xs:
                accs = []
                for act_id, _, _ in actions:
                    accs += [acc for smp, acc in running_accuracies[act_id] if smp==sample]
                ys_avg += [np.max(accs)]
                ys_err += [np.sqrt(np.std(accs))/len(accs)]

            plt.plot(xs, ys_avg,
                         marker='o',
                         ms=12,
                         markeredgecolor='k',
                         markeredgewidth='3',
                         lw=3,
                         c=colors[i],
                         label= exp_dictionary[key]) # "Maximum accuracy - " + exp_dictionary[key])
    plt.legend(fontsize=20)

    if save:
        filename = '{}-sample_number_accuracy.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()


def sample_confidence_figure(data, show=False, save=False):
    plt.close('all')
    fig = plt.figure(0, figsize=(13, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.xlabel('$Number\ of\ training\ samples$', fontsize=22)
    plt.ylabel('$Confidence\ \zeta_m$', fontsize=22)

    if 'results_folder' in data.keys():
        folder = data['results_folder']
        actions = data['actions']
        running_ba = data['running_ba']
        xs = [sample for sample, _ in running_ba[list(running_ba.keys())[0]]]

        ys_avg = []
        ys_err = []
        for sample in xs:
            bas = []
            for act_id, _, _ in actions:
                bas += [(1 - 1 / (2 + np.exp(-ba))) for smp, ba in running_ba[act_id] if smp==sample]
            ys_avg += [np.average(bas)]
            ys_err += [np.sqrt(np.std(bas))/len(bas)]

        plt.errorbar(xs, ys_avg, yerr=ys_err, uplims=True, lolims=True, elinewidth=3, lw=3, c='k')
    else:
        folder = data[list(data.keys())[0]]['results_folder']
        colors = ['#000dff', '#26d100', '#a31000', '#00a2ad']
        for i, key in enumerate(data.keys()):
            actions = data[key]['actions']
            running_ba = data[key]['running_ba']
            xs = [sample for sample, _ in running_ba[list(running_ba.keys())[0]]]

            ys_avg = []
            ys_err = []
            for sample in xs:
                bas = []
                for act_id, _, _ in actions:
                    bas += [(1 - 1 / (2 + np.exp(-ba))) for smp, ba in running_ba[act_id] if smp==sample]
                ys_avg += [np.average(bas)]
                ys_err += [np.sqrt(np.std(bas))/len(bas)]

            plt.plot(xs, ys_avg,
                     lw=3,
                     c=colors[i],
                     label="Average confidence - " + exp_dictionary[key])
            plt.errorbar(xs, ys_avg,
                         yerr=ys_err,
                         solid_capstyle='projecting',
                         c=colors[i],
                         capsize=15,
                         elinewidth=3,
                         lw=3,
                         label="Error $\\frac{\sqrt{\sigma}}{n}$ - " + exp_dictionary[key])
    plt.legend(fontsize=20)


    if save:
        filename = '{}-sample_number_confidence.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()


def confidence_accuracy_figure(data, show=False, save=False):
    plt.close('all')
    fig = plt.figure(0, figsize=(13, 10))
    ax = fig.add_subplot(111)
    ax.yaxis.set_tick_params(labelsize=14)
    ax.xaxis.set_tick_params(labelsize=14)
    plt.xlabel('$Number\ of\ training\ samples$', fontsize=22)
    plt.ylabel('$Accuracy$', fontsize=22)
    actions = data['actions']
    running_accuracies = data['running_accuracies']
    xs = [sample for sample, _ in running_accuracies[list(running_accuracies.keys())[0]]]

    ys_avg = []
    ys_err = []
    for sample in xs:
        accs = []
        for act_id, _, _ in actions:
            accs += [acc for smp, acc in running_accuracies[act_id] if smp==sample]
        ys_avg += [np.average(accs)]
        ys_err += [np.sqrt(np.std(accs))/len(accs)]

    plt.errorbar(xs, ys_avg,
                 yerr=ys_err,
                 uplims=True,
                 lolims=True,
                 elinewidth=3,
                 solid_capstyle='projecting',
                 capsize=15,
                 lw=3,
                 label="Average diagnostic accuracy",
                 c='k')

    plt.legend(fontsize=20)

    if save:
        filename = '{}-sample_number_complexity.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()


def generate_raw_data_figure(data=None, action=0, rank=0, folder="", show=False, save=True,
                            figure=None, grdspc=None, colorbar=False):
    act = action[0]
    original_belief_data = data["original_belief_data"][act]

    sample_idx = 1

    if figure is not None:
        fig = figure
        grid = gridspec.GridSpecFromSubplotSpec(1, len(list(original_belief_data.keys()))*10-1, subplot_spec=grdspc)
    else:
        fig = plt.figure(figsize=(13, 10), constrained_layout=True)
        grid = gridspec.GridSpec(ncols=len(list(original_belief_data.keys())),
                                 nrows=1,
                                 figure=fig)
    # fig.subplots_adjust(hspace=0.4, wspace=0.4)
    min_glob = None
    max_glob = None
    for i, obj in enumerate(sorted(original_belief_data.keys())):
        min_val = np.min(original_belief_data[obj][sample_idx, :])
        max_val = np.max(original_belief_data[obj][sample_idx, :])
        if min_glob is None or min_glob > min_val:
            min_glob = min_val
        if max_glob is None or max_glob < max_val:
            max_glob = max_val
    for i, obj in enumerate(sorted(original_belief_data.keys())):
        colorbarplot_shifter = 1
        colorbar_increaser = 0
        if colorbar and i == len(list(original_belief_data.keys())) - 1:
            colorbarplot_shifter = 0
            colorbar_increaser = 2

        ax_subplot = fig.add_subplot(
            grid[:, i*10+colorbarplot_shifter:i*10+7+colorbarplot_shifter+colorbar_increaser]
        )
        ax_subplot.set_xlabel('$time$', fontsize=48)
        ax_subplot.set_ylabel('$taxel\ values$', fontsize=48)
        plt.autoscale(True)
        ax_subplot.yaxis.set_tick_params(labelsize=22)
        ax_subplot.xaxis.set_tick_params(labelsize=22)
        img = original_belief_data[obj][sample_idx, :].reshape(5, -1).T.astype(np.int64)
        # img = (img.T - np.average(img, axis=1)).T
        ax_subplot.set_title(label_dict_abbrv[obj], fontsize=30)
        ax_subplot.set_xlabel('$time\ (s)$', fontsize=32)
        ax_subplot.set_ylabel('$taxels$', fontsize=32)
        ax_subplot.set_xticks(list(np.arange(0.5, 4., 2.)))
        ax_subplot.set_xticklabels([str(sec) for sec in list(np.arange(0.5, 4., 2.))])
        ax_subplot.set_yticks(list(range(7)))
        ax_subplot.set_yticklabels(['t' + str(tax) for tax in range(7)])
        # fig.add_subplot(ax_subplot)
        im = ax_subplot.imshow(img, cmap='hot', interpolation='bilinear', vmin=min_glob, vmax=max_glob)

        if colorbar and i == len(list(original_belief_data.keys()))-1:
            divider = make_axes_locatable(ax_subplot)
            cax = divider.append_axes("right", size="20%", pad=0.5)
            cbar = plt.colorbar(im, cax=cax)
            cbar.ax.get_yaxis().labelpad = 15
            cbar.ax.tick_params(labelsize=26)
            cbar.ax.set_ylabel('$normalized\ sensor\ values$', rotation=270, fontsize=32, labelpad=30)

    if save:
        filename = '{}{}_{}-raw_plots.png'.format(RESULTS_FOLDER, rank, action[0])
        fig.savefig(filename, bbox_inches="tight", dpi=300)
    if show:
        plt.show()

    return True


def generate_motion_param_comparison_figure2(data, save_key="", show=False, save=False):
    plt.close('all')
    action_accuracies = data['action_accuracies']
    actions = data['actions']
    folder = data['results_folder']

    fig = plt.figure(figsize=(13, 10))
    grid = gridspec.GridSpec(ncols=6, nrows=1, figure=fig, wspace=.5, hspace=.5)
    fig.add_subplot(111, frameon=False)
    plt.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)
    plt.xlabel('$Action\ Parameters$', fontsize=28, labelpad=40)
    plt.ylabel('$Accuracy$', fontsize=28, labelpad=20)

    params = [(0, (1, 2), "$\omega^Z$"),
              (1, (1, 3), "$\omega^{Rx}$"),
              (2, (1, 4), "$\omega^{Ry}$"),
              (3, (2, 2), "$A^{Z}$"),
              (4, (2, 3), "$A^{Rx}$"),
              (5, (2, 4), "$A^{Ry}$")]
    augmented_params = []

    params_values = np.zeros((len(params), len(actions)))
    for i, idxesm, param in params:
        for j, action in enumerate(actions):
            params_values[i, j] = action[idxesm[0]][idxesm[1]]
        vals = (np.unique(params_values[i, :]),)
        augmented_params += [(params[i] + vals)]

    max_acc = 0
    subplots = []
    for i, idxes, param, vals in augmented_params:
        if i == 0:
            subplots += [fig.add_subplot(grid[:, i])]
            subplots[-1].yaxis.set_tick_params(labelsize=18)
        else:
            subplots += [fig.add_subplot(grid[:, i], sharey=subplots[-1])]
        subplots[-1].set_xlabel(param, fontsize=22)

        box_data = []
        for j, val in enumerate(vals):
            accuracies = []
            for act, accuracy in action_accuracies:
                action = [action for action in actions if action[0] == act][0]
                if action[idxes[0]][idxes[1]] == val:
                    accuracies += [accuracy]
                    if accuracy > max_acc:
                        max_acc = accuracy
            box_data += [accuracies]
        subplots[-1].boxplot(box_data, widths=0.5)

        if i == 3:
            subplots[-1].set_xticklabels(["{:.1f}".format(x*1000) for x in vals])
        else:
            subplots[-1].set_xticklabels(["{:.1f}".format(x) for x in vals])

        subplots[-1].xaxis.set_tick_params(labelsize=18)
        subplots[-1].get_yaxis().set_ticks(np.arange(0., 1., .1))
        subplots[-1].set_yticklabels(["{:.1f}".format(x) for x in np.arange(0., 1., .1)])

    for subplot in subplots:
        subplot.set_ylim([0, max_acc+.05])

    if save:
        filename = '{}{}-action_comparison2.png'.format(RESULTS_FOLDER, save_key)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return None


def generate_pca_figure(data_bayesian=None, data_systematic=None, show=False, save=False):
    actions = data_bayesian['actions']
    folder = data_bayesian['results_folder']
    original_belief_data = data_systematic['original_belief_data']

    ranked_idxs_to_extract = list(np.ceil(np.arange(0, len(actions), len(actions)/3)).astype(np.int))+[len(actions)-1]
    for i in ranked_idxs_to_extract:
        generate_pca_histogram(actions=actions,
                               original_belief_data=original_belief_data,
                               rank=i,
                               folder=folder,
                               show=show,
                               save=save)
        plt.close('all')
    return True


def generate_figure_10(exp_number=None, data_bayesian=None, data_systematic=None, figure=None, legend=False,
                       grdspc=None,  show=False, save=True):
    plt.close('all')
    cut_off_idx = min(len(data_bayesian["best_accuracies"]), len(data_systematic["best_accuracies"])) + 1
    bayesian_accuracies = (np.array(data_bayesian["best_accuracies"][:cut_off_idx])*100).astype(np.int)
    systematic_accuracies = (np.array(data_systematic["best_accuracies"][:cut_off_idx])*100).astype(np.int)
    folder = data_bayesian['results_folder']

    if figure is not None:
        fig = figure
        ax = fig.add_subplot(grdspc)
    else:
        fig = plt.figure(1, figsize=(13, 10))
        ax = fig.add_subplot(111)

    if exp_number is not None:
       ax.set_title("{}\n\n".format(exp_dictionary[exp_number]), loc="center", fontsize=42, fontweight='bold')
    ax.set_xlabel('$Palpation\ iterations$', fontsize=34)  # \ (1unit=1action&4objects)
    ax.set_ylabel('$Accuracy\ (\%)$', fontsize=34)

    ax.plot(systematic_accuracies,
             linewidth=6,
             linestyle='--',
             color='k',
             label="Grid Search",
             marker="o",
             alpha=.7)
    ax.plot(bayesian_accuracies,
             linewidth=6,
             color='r',
             marker="o",
             label="Bayesian Exploration")

    # add converted iterations in hours
    ax2 = ax.twiny()
    ax1Xs = ax.get_xticks()
    upperXTickMarks = ["{:.2f}".format(x*.0302+.20) for x in ax1Xs]
    for i, tick in enumerate(upperXTickMarks):
        upperXTickMarks[i] = "{}:{}h".format(tick.split('.')[0], str(int(float(tick.split('.')[1])*60/100)))
    ax2.set_xticks(ax1Xs)
    ax2.set_xbound(ax.get_xbound())
    ax2.set_xticklabels(upperXTickMarks)
    # title = ax.set_title("$Palpation\ exploration\ time$", fontsize=28)
    ax.text(0.5, 1.18, "$Palpation\ exploration\ time$",
             horizontalalignment='center',
             fontsize=34,
             transform=ax2.transAxes)
    ax.yaxis.set_tick_params(labelsize=20)
    ax.xaxis.set_tick_params(labelsize=20)
    ax2.xaxis.set_tick_params(labelsize=20)

    if legend:
        handles, labels = ax.get_legend_handles_labels()
        handles = [copy.copy(handle) for handle in handles]
        [handle.set_linewidth(10) for handle in handles]
        [handle.set_alpha(1) for handle in handles]
        ax.legend(handles=handles, labels=labels, fontsize=42, loc='upper right',
                  bbox_to_anchor=(2.8, -.25), ncol=2, fancybox=True)

    if save:
        filename = '{}fig10.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

    return fig


def generate_figure_4C(data, show=False, save=True):

    plt.close('all')
    folder = data[list(data.keys())[0]]['results_folder']

    fig = plt.figure(figsize=(55, 25), constrained_layout=True)
    ax = fig.add_subplot(111, frameon=False)
    ax.tick_params(labelcolor='none', top=False, bottom=False, left=False, right=False)

    grid = gridspec.GridSpec(ncols=20, nrows=20, figure=fig)  # wspace=.8, hspace=.1
    grid.update(wspace=0.25, hspace=0.4)  # set the spacing between axes.

    y_coord_labels = ["Rank 0\nTraining\nPhantom 1", "Rank 0\nTraining\nPhantom 2", "Rank 0\nAbdominal\nPhantom"]
    label_grids = [grid[3:8, :], grid[9:14, :], grid[15:, :]]
    # titles = ["actions", "Ph-1", "Ph-2", "Abdominal"]
    titles = ["actions", "Training Phantom 1", "Training Phantom 2", "Abdominal Phantom"]
    for i in range(len(label_grids)):
        ax_main = fig.add_subplot(label_grids[i], frameon=False)
        ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
        ax_main.set_ylabel(y_coord_labels[i], fontsize=58, labelpad=15, rotation=90)

    ax_main = fig.add_subplot(grid[:, 1:4], frameon=False)
    ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
    ax_main.set_title(titles[0], fontsize=58, fontweight='bold', pad=-150)
    for i in range(1, 4):
        ax_main = fig.add_subplot(grid[:, (i)*5:(i)*5+4], frameon=False)
        ax_main.tick_params(labelcolor='none', top=False, bottom=False, right=False, left=False)
        ax_main.set_title(titles[i], fontsize=58, fontweight='bold', pad=-150)

    handles, labels = (None, None)
    phantoms = [('exploration_training_phantom_2', "Training Phantom 1"), ('exploration_training_phantom_1', "Training Phantom 2"), ('abdominal_phantom', "Test Phantom")]

    generate_motion_profile(action=[action for action in data['exploration_training_phantom_2']['actions']
                                    if action[0] == data['exploration_training_phantom_2']['unbiased_ranked_actions'][0]][0],
                            data=data['exploration_training_phantom_2'],
                            show=show,
                            legend=False,
                            no_title=True,
                            save=False,
                            figure=fig,
                            grdspc=grid[3:8, 1:4])
    generate_motion_profile(action=[action for action in data['exploration_training_phantom_2']['actions']
                                    if action[0] == data['exploration_training_phantom_1']['unbiased_ranked_actions'][0]][0],
                            data=data['exploration_training_phantom_1'],
                            show=show,
                            legend=False,
                            no_title=True,
                            save=False,
                            figure=fig,
                            grdspc=grid[9:14, 1:4])
    generate_motion_profile(action=[action for action in data['exploration_training_phantom_2']['actions']
                                    if action[0] == data['abdominal_phantom']['unbiased_ranked_actions'][0]][0],
                            data=data['abdominal_phantom'],
                            show=show,
                            legend=False,
                            no_title=True,
                            save=False,
                            figure=fig,
                            grdspc=grid[15:, 1:4])

    all_handles = []
    all_labels = []
    for i, (key, label) in enumerate(phantoms):
        print("generated subfig {}-{} {}, out of {} done".format(1, i, i*3+2, 12))

        # bit of code to fix the problem that the palpation on abdominal phantom has only beset 20 acts from ph2
        action_found = False
        idx = 0
        while not action_found:
            acts = [action for action in data[key]['actions'] if action[0] == data['exploration_training_phantom_2']['unbiased_ranked_actions'][idx]]
            if len(acts) != 0:
                act = acts[0]
                action_found = True
            else:
                idx += 1
        generate_belief_state(
                              data=data[key],
                              action=act,
                              folder=folder,
                              show=show,
                              legend=False,
                              save=False,
                              figure=fig,
                              grdspc=grid[3:8, (i+1)*5:(i+1)*5+4])
        print("generated subfig {}-{} {}, out of {} done".format(2, i, i*3+3, 12))
        generate_belief_state(
                              data=data[key],
                              action=[action for action in data[key]['actions']
                                      if action[0] == data['exploration_training_phantom_1']['unbiased_ranked_actions'][0]][0],
                              folder=folder,
                              show=show,
                              legend=False,
                              save=False,
                              figure=fig,
                              grdspc=grid[9:14, (i+1)*5:(i+1)*5+4])
        print("generated subfig {}-{} {}, out of {} done".format(2, i, i*3+3, 12))
        handles, labels = generate_belief_state(
                              data=data[key],
                              action=[action for action in data[key]['actions']
                                      if action[0] == data['abdominal_phantom']['unbiased_ranked_actions'][0]][0],
                              folder=folder,
                              show=show,
                              legend=i==0,
                              save=False,
                              figure=fig,
                              grdspc=grid[15:, (i+1)*5:(i+1)*5+4])
        all_handles += handles
        all_labels += labels
        print("generated subfig {}-{} {}, out of {} done".format(2, i, i*3+3, 12))

    handles_curves = []
    labels_curves = []
    handles_scatter = []
    labels_scatter = []

    # inverse the labels for visualization
    def fun(label):
        num = ''.join([k for k in label if k.isdigit()])
        try:
            return int(num)
        except:
            return 0

    for lbl in sorted(label_dict.keys(), key=lambda x: fun(x), reverse=True):
        curve_found = False
        scatter_found = False
        for i, handle in enumerate(all_handles):
            if all_labels[i] == label_dict[lbl]:
                if handle._linestyle == '-' and not curve_found:
                    handles_curves += [handle]
                    labels_curves += [all_labels[i]]
                    curve_found = True
                if handle._linestyle == 'None' and not scatter_found:
                    handles_scatter += [handle]
                    labels_scatter += [all_labels[i]]
                    scatter_found = True
            if curve_found and scatter_found:
                break

    handles = []
    labels = []
    for i in range(len(handles_curves)):
        handles += [handles_curves[i]]
        handles += [handles_scatter[i]]
        labels += [labels_curves[i]]
        labels += [labels_scatter[i]]

    ax.legend(handles=handles, labels=labels, fontsize=48,
              loc='lower center', bbox_to_anchor=(.48, -0.25), ncol=5, fancybox=True)

    if save:
        filename = '{}Figure_complexity.png'.format(RESULTS_FOLDER)
        fig.savefig(filename, bbox_inches="tight", dpi=150)
        plt.close('all')
    return fig



if __name__ == "__main__":

    # GENERATE ALL FIGURES

    # x, y, z, rx, ry, rz -- > motion parameters to specify action: see paper
    eta_min = np.array([0., 0., .5, 1., 1., 0.])
    eta_max = np.array([0., 0., 2., 3., 3., 0.])
    A_min = np.array([0., 0., .001, -np.deg2rad(10), -np.deg2rad(10), 0.])
    A_max = np.array([0., 0., .005, np.deg2rad(10), np.deg2rad(10), 0.])  # 80 mm depth

    # Generate 'number_of_actions' intervals for each of the min-max ranges given -- combinatorial!
    actions = get_action_profile(eta_min, eta_max, A_min, A_max, number_of_actions=2)

    environment = {
        '10mil': [(.0081, -.4398), (.0447, -.4466), (.0790, -.4501), (.1239, -.4598)],
        '5mil': [(-.0062, -.5065), (.0355, -.5105), (.0758, -.5159), (.1211, -.5200)],
        'na': [(.0019, -.3715), (.0473, -.3750), (.0894, -.3825), (.1347, -.3851)]
    }

    exp_bayesian_data = {}
    exp_systematic_data = {}

    time_limit = 6
    for folder_name in exp_dictionary.keys():
        brain = BayesianBrain(
            environment=environment,
            actions=actions,
            samples_per_pdf=20,
            sensing_resolution=time_limit,
            verbose=False,
            dimensionality_reduction=1,
            training_ratio=.7,
            experiment_name=folder_name,
        )

        # simulate run of bayesian experiments from logged data
        exp_bayesian_data[folder_name] = brain.run_experiment(bayesian=True,
                                                    initial_sample_number=1,
                                                    test_sample_number=9,
                                                    show=False,
                                                    save=False)
        # simulate run of systematic experiments from logged data
        exp_systematic_data[folder_name] = brain.run_experiment(bayesian=False,
                                                      initial_sample_number=1,
                                                      test_sample_number=9,
                                                      show=False,
                                                      save=False)
        # create folder to dump results
    folder_create(RESULTS_FOLDER, exist_ok=True)

    # axis_complexity_figure(exp_systematic_data['complexity_demonstration'], show=False, save=True)
    #
    # sample_accuracy_figure(exp_systematic_data, show=False, save=True)
    # sample_confidence_figure(exp_systematic_data, show=False, save=True)
    # confidence_accuracy_figure(exp_systematic_data['complexity_demonstration'], show=False, save=True)
    # generate_figure_4C(exp_systematic_data, show=False, save=True)
    # generate_motion_param_comparison_figure2(exp_systematic_data['exploration_training_phantom_2'],
    #                                          save_key='_training_phantom_2',
    #                                          show=False,
    #                                          save=True)
    # generate_motion_param_comparison_figure2(exp_systematic_data['exploration_training_phantom_1'],
    #                                          save_key='_training_phantom_1',
    #                                          show=False,
    #                                          save=True)
    # generate_motion_param_comparison_figure2(exp_systematic_data['abdominal_phantom'],
    #                                          save_key='_abdominal_phantom',
    #                                          show=False,
    #                                          save=True)
    #
    # generate_figure_3(data=exp_bayesian_data['exploration_training_phantom_2'],
    #                   save_key='_training_phantom_2',
    #                   show=False,
    #                   save=True)
    # generate_figure_3(data=exp_bayesian_data['exploration_training_phantom_1'],
    #                   save_key='_training_phantom_1',
    #                   show=False,
    #                   save=True)
    # generate_figure_3(data=exp_systematic_data['abdominal_phantom'],
    #                   save_key='_abdominal_phantom',
    #                   show=False,
    #                   save=True)
    # generate_figure_4(
    #     all_bayesian_data=exp_bayesian_data,
    #     all_systematic_data=exp_systematic_data,
    #     show=False,
    #     save=True
    # )
    # generate_figure_5(
    #     all_bayesian_data=exp_bayesian_data,
    #     all_systematic_data=exp_systematic_data,
    #     show=False,
    #     save=True
    # )
    generate_figure_6(
        data=exp_bayesian_data['exploration_training_phantom_2'],
        save_key='training_phantom_2_',
        show=False,
        save=True
    )
    generate_figure_6(
        data=exp_bayesian_data['exploration_training_phantom_1'],
        save_key='training_phantom_1_',
        show=False,
        save=True
    )
    generate_figure_6(
        data=exp_bayesian_data['abdominal_phantom'],
        save_key='abdominal_phantom_',
        show=False,
        save=True
    )

