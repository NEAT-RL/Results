import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series, DataFrame
from numpy import nan as NA
import glob
import matplotlib.pyplot as plt
import numpy as np
import os

if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    ferms = glob.glob(local_dir + '/data/EM_THEANO/csv_files/*.csv')
    #
    # ferms = glob.glob('csv_files/agent*.csv')
    # print(ferms)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for ferm in ferms:
        # define the dataframe
        # data = pd.read_csv(ferm, header=None, usecols=[0, 1])
        headers = ["iteration", "average_steps", "average_reward"]
        data_no_headers = pd.read_csv(ferm, names=headers)
        ax.plot(data_no_headers['iteration'], data_no_headers['average_steps'])

    plt.xlabel('iteration')
    plt.ylabel('average_steps/reward')
    plt.title('Average steps/rewards per iteration of EM on Cartpole using Theano')
    plt.show()
    # plt.savefig('EM_STATE_TRANSITIONS.png', bbox_inches='tight')
