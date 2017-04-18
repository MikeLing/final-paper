import numpy as np
import csv
#from matplotlib.pyplot import plot
from scipy.sparse import coo_matrix
import scipy.sparse as sps
import pdb


"""make a checkpoint for time slice

:param step the step of each snapshot
:param row each row of the mataData and it looks like
        node1, node2, linkstart, linkend, connect time, totally connected time

"""


def time_slicer(steps, min_time, max_time):
    # how many snapshot we are going to have
    looper = (max_time - min_time)/steps
    checkpoints = [set([min_time + steps * l, min_time + steps * (l+1)]) for l in range(0, looper)]
    return checkpoints


"""
    Weight cacluation
"""


def link_weight(duration, k):
    return np.exp(duration/k)


"Generate spare martix"
"""
One thing must remember is that node pair in the data has connected, so we don't need
worried about take node pair without connected into account.
"""


def spare_martix_generator(slice_start, slice_end, mataData):
    row = []
    col = []
    data = []
    k = 160
    for i in mataData:
        duration = (i[3] if i[3] < slice_end else slice_end) - (i[2] if i[2] > slice_start else slice_start)
        #pdb.set_trace()
        # sum weight up if we already have that data
        if i[0] in row and col[row.index(i[0])] == i[1]:
            data[row.index(i[0])] += link_weight(duration, k)
        else:
            row.append(i[0])
            col.append(i[1])
            data.append(link_weight(duration, k))

    m = coo_matrix((data, (row, col)), shape=(21780, 21780))

    print m


if __name__ == "__main__":
    # load the dataset
    mataData = np.genfromtxt("../MSNs data/INFOCOM06.txt")

    # time range of the dataset
    t_min = min(mataData[:, 2])
    t_max = max(mataData[:, 3])
    time_range = t_max - t_min

    step = 0
#    checkpoints = time_slicer(steps=step, min_time=t_min, max_time=t_max)
    spare_martix_generator(t_min, t_max, mataData)

