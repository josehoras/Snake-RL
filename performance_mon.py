import matplotlib.pyplot as plt
import numpy as np


def update_performance(performance, number, level_moves):
    performance['score'].append(number - 1)
    smooth_score = np.sum(performance['score'][-100:]) / len(performance['score'][-100:])
    performance['smooth_score'].append(smooth_score)
    performance['moves'].append(level_moves)
    smooth_move = np.sum(performance['moves'][-100:]) / len(performance['moves'][-100:])
    performance['smooth_moves'].append(smooth_move)
    return performance


def plot_screens(screens):
    rows = 3
    cols = 3
    n = rows * cols
    fig = plt.figure(figsize=(cols * 2.25, rows * 2.25), dpi=100)

    for i in range(n):
        plt.subplot(rows, cols, i + 1)  # sets the number of feature maps to show on each row and column
        plt.title(str(i), fontsize=10)
        # plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        plt.imshow(screens[i], cmap='Greys')
        # see_input = screens[i][screens[i]!=0]
        # print(see_input)
    plt.tight_layout()
    fig.savefig('results/screens_plot.png')
    plt.close("all")
    # plt.show()


def plot_performance(performance, file_base_name):
    fig = plt.figure(figsize=(7, 7), dpi=100)
    length = len(performance['moves'])
    lwidth = 1 - min(length//1000, 9) * 0.1
    msize = 2 - min(length//1000, 10) * 0.1
    plt.subplot(2, 1, 1)
    plt.title("total moves", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(which='major', axis='y', linestyle=':', linewidth=0.5)
    plt.plot(performance['moves'], linewidth=0, marker=".", markersize=msize)
    plt.plot(performance['smooth_moves'])
    plt.subplot(2, 1, 2)
    plt.title("score", fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(which='major', axis='y', linestyle=':', linewidth=0.5)
    plt.plot(performance['score'], linewidth=lwidth)#, marker=",", markersize=1)  # , color='g')
    plt.plot(performance['smooth_score'])
    # ax1 = fig.add_subplot(111)
    # ax2 = ax1.twinx()
    # # plt.plot(performance['score'])
    # ax1.plot(performance['moves'])
    # ax1.plot(performance['smooth_moves'])
    # ax2.plot(performance['score'], color='g')
    # ax2.set_ylim([0, 10])
    plt.tight_layout()
    fig.savefig(file_base_name + '_perf.png')
    plt.close("all")
    # plt.show()


def plot_value(value, x, y, file_name='', alpha='', gamma='', r='', it=''):
    def probs(matrix):
        # print(matrix)
        return np.exp(matrix) / np.sum(np.exp(matrix), axis=1).reshape(-1, 1)

    def print_probs(values, x, y):
        print("Probs on %i, %i: " % (x, y))
        print("\t\t\t\t   Left   Right  Up     Down   Nothing")
        for i in range(4):
            print("  Going %s: \t%s" % (directions[i], values[i]))

    actions = ['left', 'right', 'up', 'down', 'no action']
    directions = ['left', 'right', 'up', 'down']
    fig = plt.figure(figsize=(7, 7), dpi=100)

    value_plane = value[:, :, x, y]

    np.set_printoptions(formatter={'float': ' {: 0.2f}'.format})
    # print_probs(probs(value_plane[0, 10]), 0, 10)
    # print_probs(probs(value_plane[1, 10]), 1, 10)
    # print_probs(probs(value_plane[5, 10]), 5, 10)
    # print_probs(probs(value_plane[6, 10]), 6, 10)
    # print_probs(probs(value_plane[7, 10]), 7, 10)
    # print_probs(probs(value_plane[8, 10]), 8, 10)

    d = 0
    fig.suptitle("On direction: " + directions[d], fontsize=16)
    for i in range(5):
        plt.subplot(2, 3, i + 1)
        plt.title("a: " + actions[i], fontsize=14)
        plt.xticks([])
        plt.yticks([])

        plt.imshow(np.swapaxes(value_plane[:, :, d, i], 0, 1),
                   interpolation='nearest', cmap='gist_heat')
        plt.colorbar()
    if alpha != '':
        fig.text(0.7, 0.35, 'alpha = {}'.format(alpha), fontsize=16)
        fig.text(0.7, 0.3, 'gamma = {}'.format(gamma), fontsize=16)
        fig.text(0.7, 0.25, 'r = {}'.format(r), fontsize=16)
        fig.text(0.7, 0.2, 'it = {}'.format(it), fontsize=16)
    fig.tight_layout(pad=2.5, h_pad=0.25, w_pad=0.25)
    fig.savefig(file_name)
    plt.close("all")


def plot_probs(model, x, y, file_name=''):

    def e_expected(state, e):
        p = np.full_like(state, e * (1/len(state)), dtype=float)
        for i in range(p.shape[0]):
            for j in range(p.shape[1]):
                for k in range(p.shape[2]):
                    p[i][j][k][np.argmax(state[i][j][k])] += 1 - e
        return p

    directions = ['left', 'right', 'up', 'down']
    fig = plt.figure(figsize=(7, 7), dpi=100)
    value_plane = model.q[:, :, x, y]
    if model.policy == model.softmax_policy:
        # probs_plane = np.exp(value_plane) / np.sum(np.exp(value_plane), axis=3).reshape(20, 20, 4, 1)
        probs_plane = value_plane
    else:
        probs_plane = e_expected(value_plane, model.e)


    # d = random.randint(0, 3)
    d = 0
    ds = [{0: [0, 1, 4], 1: [2], 2: [3]},
          {0: [0, 1, 4], 1: [2], 2: [3]},
          {0: [2, 3, 4], 1: [0], 2: [1]},
          {0: [2, 3, 4], 1: [0], 2: [1]}]
    bctions = [['keep going', 'up', 'down'],
               ['keep going', 'up', 'down'],
               ['keep going', 'left', 'right'],
               ['keep going', 'left', 'right']]

    # fig.suptitle("On direction: " + directions[d], fontsize=16)
    axs = []
    rows = 4
    for d in range(rows):
        for i in range(3):
            # axs[i] = plt.subplot(2, 3, i + 1)
            axs.append(fig.add_subplot(rows, 3, (i + 1) + d * 3))

            plt.title("a: " + bctions[d][i], fontsize=14)
            plt.xticks([])
            plt.yticks([])

            probs_sum = np.zeros_like(np.swapaxes(probs_plane[:, :, d, i], 0, 1))
            # print(ds[0])
            for j in ds[d][i]:
                probs_sum += np.swapaxes(probs_plane[:, :, d, j], 0, 1)

            im = axs[-1].imshow(probs_sum, interpolation='nearest', cmap='viridis')#, vmin=0, vmax=1)
            # if i==2:
            #     cbar = fig.colorbar(im, ax=ax, shrink=0.62, format='%.1f') #,use_gridspec=True

    if model.alpha != '' and rows < 3:
        fig.text(0.7, 0.35, 'alpha = {}'.format(model.alpha), fontsize=16)
        fig.text(0.7, 0.3, 'gamma = {}'.format(model.gamma), fontsize=16)
        fig.text(0.7, 0.25, 'r = {}'.format(model.r), fontsize=16)
        fig.text(0.7, 0.2, 'it = {}'.format(len(model.performance['moves'])), fontsize=16)
    # fig.colorbar(cm.ScalarMappable(cmap='viridis'))
    # fig.tight_layout(pad=2.5, h_pad=0.25, w_pad=0.25)
    fig.tight_layout()
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax)
    fig.subplots_adjust(right=1.05)
    fig.colorbar(im, ax=axs, shrink=0.54, pad=0.02, aspect=10, format='%.1f')
    fig.savefig(file_name + '_prob.png')
    plt.close("all")
