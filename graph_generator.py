import pandas as pd
import glob
import matplotlib.pyplot as plt
import os

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    plt.style.use('ggplot')
    fig, axes = plt.subplots(ncols=2, nrows=3)
    ax11, ax12, ax21, ax22, ax31, ax32 = axes.ravel()

    ferms = glob.glob(local_dir + '/data/EM_ALGORITHMS/data/cartpole/power_po/**/*.csv', recursive=True)
    data_frames = []
    for ferm in ferms:
        data_no_headers = pd.read_csv(ferm)
        data_frames.append(data_no_headers)

    df = pd.concat([data_frame for data_frame in data_frames])
    average_results = df.groupby(df.index).mean()
    ax11.plot(average_results['Iteration'], average_results['AverageReturn'])
    ax11.fill_between(average_results['Iteration'],
                    df.groupby(df.index).min()['AverageReturn'],
                    df.groupby(df.index).max()['AverageReturn'],
                    alpha=0.2, facecolor='#089FFF')
    ax11.set_ylabel('Average Reward')
    ax11.set_title('CartPole-v0')

    ferms = glob.glob(local_dir + '/data/EM_ALGORITHMS/data/mountaincar/power_po/**/*.csv', recursive=True)
    data_frames = []
    for ferm in ferms:
        data_no_headers = pd.read_csv(ferm)
        data_frames.append(data_no_headers)

    df = pd.concat([data_frame for data_frame in data_frames])
    average_results = df.groupby(df.index).mean()
    ax12.plot(average_results['Iteration'], average_results['AverageReturn'])
    ax12.fill_between(average_results['Iteration'],
                     df.groupby(df.index).min()['AverageReturn'],
                     df.groupby(df.index).max()['AverageReturn'],
                     alpha=0.2, facecolor='#089FFF')
    ax12.margins(0)
    ax12.set_title('MountainCarExtraLong-v0')


    # Power Po Variant
    ferms = glob.glob(local_dir + '/data/EM_ALGORITHMS/data/cartpole/power_po_variant/**/*.csv', recursive=True)
    data_frames = []
    for ferm in ferms:
        data_no_headers = pd.read_csv(ferm)
        data_frames.append(data_no_headers)

    df = pd.concat([data_frame for data_frame in data_frames])
    average_results = df.groupby(df.index).mean()
    ax21.plot(average_results['Iteration'], average_results['AverageReturn'])
    ax21.fill_between(average_results['Iteration'],
                     df.groupby(df.index).min()['AverageReturn'],
                     df.groupby(df.index).max()['AverageReturn'],
                     alpha=0.2, facecolor='#089FFF')
    ax21.set_ylabel('Average Reward')


    ferms = glob.glob(local_dir + '/data/EM_ALGORITHMS/data/mountaincar/power_po_variant/**/*.csv', recursive=True)
    data_frames = []
    for ferm in ferms:
        data_no_headers = pd.read_csv(ferm)
        data_frames.append(data_no_headers)

    df = pd.concat([data_frame for data_frame in data_frames])
    average_results = df.groupby(df.index).mean()
    ax22.plot(average_results['Iteration'], average_results['AverageReturn'])
    ax22.fill_between(average_results['Iteration'],
                     df.groupby(df.index).min()['AverageReturn'],
                     df.groupby(df.index).max()['AverageReturn'],
                     alpha=0.2, facecolor='#089FFF')
    ax22.margins(0)


    # Power Po Gradient
    ferms = glob.glob(local_dir + '/data/EM_ALGORITHMS/data/cartpole/power_po_gradient/**/*.csv', recursive=True)
    data_frames = []
    for ferm in ferms:
        data_no_headers = pd.read_csv(ferm)
        data_frames.append(data_no_headers)

    df = pd.concat([data_frame for data_frame in data_frames])
    average_results = df.groupby(df.index).mean()
    ax31.plot(average_results['Iteration'], average_results['AverageReturn'])
    ax31.fill_between(average_results['Iteration'],
                     df.groupby(df.index).min()['AverageReturn'],
                     df.groupby(df.index).max()['AverageReturn'],
                     alpha=0.2, facecolor='#089FFF')
    ax31.set_ylabel('Average Reward')
    ax31.set_xlabel('Generations')

    ferms = glob.glob(local_dir + '/data/EM_ALGORITHMS/data/mountaincar/power_po_gradient/**/*.csv', recursive=True)
    data_frames = []
    for ferm in ferms:
        data_no_headers = pd.read_csv(ferm)
        data_frames.append(data_no_headers)

    df = pd.concat([data_frame for data_frame in data_frames])
    average_results = df.groupby(df.index).mean()
    ax32.plot(average_results['Iteration'], average_results['AverageReturn'])
    ax32.fill_between(average_results['Iteration'],
                     df.groupby(df.index).min()['AverageReturn'],
                     df.groupby(df.index).max()['AverageReturn'],
                     alpha=0.2, facecolor='#089FFF')
    ax32.margins(0)
    ax32.set_xlabel('Generations')


    """
    For each ferm in ferms. I want to 

    """
    # all_files = pd.concat([pd.read_csv(ferm) for ferm in ferms[:2]])  # you can easily put your 1000 files here
    # result = {}
    # for i in range(500):  # 3 being number of generations
    #     result[i] = all_files[i::500].mean()
    # result_df = pd.DataFrame(result)
    # ax.plot(result_df.T['Iteration'], result_df.T['AverageReturn'])

    # plt.xlabel('Iteration')
    # plt.ylabel('Average reward')
    # plt.title('Average rewards on CartPole-v0 using EM-Discretised Algorithm')
    plt.show()
    # plt.savefig('EM_STATE_TRANSITIONS.png', bbox_inches='tight')



