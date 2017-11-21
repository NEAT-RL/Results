import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    plt.style.use('ggplot')

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    # headers = ["Iteration", "Steps", "AverageReturn"]

    ferms = glob.glob(local_dir + '/data/data_13_OCT/EM_ALGORITHMS/power_po_variant/cartpole/**/*.csv', recursive=True)
    data_frames = []
    for ferm in ferms:
        data_no_headers = pd.read_csv(ferm) # , names=headers)
        data_frames.append(data_no_headers)

    df = pd.concat([data_frame for data_frame in data_frames])
    average_results = df.groupby(df.index).mean()
    ax.plot(average_results.index, average_results['AverageReturn'])

    ax.set_ylabel('Average Reward')
    ax.set_title('Performance of NEAT on TimePilot-ram-v0')
    ax.set_xlabel('Iteration')


    #
    # fig, axes = plt.subplots(ncols=1, nrows=2)
    # ax1, ax2 = axes.ravel()
    # headers = ["iteration", "steps", "average_reward"]
    #
    # ferms = glob.glob(local_dir + '/NEAT_ALGORITHMS/flappybird/**/*.csv', recursive=True)
    # data_frames = []
    # for ferm in ferms:
    #     data_no_headers = pd.read_csv(ferm, names=headers)
    #     data_frames.append(data_no_headers)
    #
    # df = pd.concat([data_frame for data_frame in data_frames])
    # average_results = df.groupby(df.index).mean()
    # ax1.plot(average_results['iteration'], average_results['average_reward'])
    # # ax1.fill_between(average_results['steps'],
    # #                 df.groupby(df.steps, as_index=False).min()['average_reward'],
    # #                 df.groupby(df.steps, as_index=False).max()['average_reward'],
    # #                 alpha=0.2, facecolor='#089FFF')
    # ax1.set_ylabel('Average Reward')
    # ax1.set_title('CartPole-v0')
    # # ax1.set_xlabel('Generations')


    # ferms = glob.glob(local_dir + '/data/temp/**/*.csv', recursive=True)
    # data_frames = []
    # for ferm in ferms:
    #     data_no_headers = pd.read_csv(ferm, names=headers)
    #     data_frames.append(data_no_headers)
    #
    # df = pd.concat([data_frame for data_frame in data_frames])
    # average_results = df.groupby(df.steps, as_index=False).mean()
    # ax2.plot(average_results['steps'], average_results['average_reward'])
    # ax2.fill_between(average_results['steps'],
    #                 df.groupby(df.index).min()['average_reward'],
    #                 df.groupby(df.index).max()['average_reward'],
    #                 alpha=0.2, facecolor='#089FFF')
    # ax2.set_ylabel('Average Reward')
    # ax2.margins(0)
    # ax2.set_title('MountainCarExtraLong-v0')
    # ax2.set_xlabel('Generations')

    plt.show()
    # plt.savefig('EM_STATE_TRANSITIONS.png', bbox_inches='tight')



