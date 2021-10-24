import numpy as np


def comp_dRR(data):
    RR_s = np.zeros((len(data) - 1, 2))
    RR_s[:, 0] = data[1:]
    RR_s[:, 1] = data[:-1]
    dRR_s = np.zeros(len(data) - 1)
    #
    # % Normalization factors (normalize according to the heart rate)
    k1 = 2
    k2 = 0.5
    #
    for i in range(len(RR_s)):
        if np.sum(RR_s[i, :] < 0.500) >= 1:
            dRR_s[i] = k1 * (RR_s[i, 0] - RR_s[i, 1])
        elif np.sum(RR_s[i, :] > 1) >= 1:
            dRR_s[i] = k2 * (RR_s[i, 0] - RR_s[i, 1])
        else:
            dRR_s[i] = RR_s[i, 0] - RR_s[i, 1]

    return dRR_s


def comput_Radius(RR_intervals):
    radius = 0
    bins = 10 // 2
    if len(RR_intervals) > 3:
        # %Compute dRR intervals series
        dRR_in = comp_dRR(RR_intervals)
        dRR_in = np.int64(np.round(dRR_in * 250))

        if len(dRR_in) > 3:
            dRR = np.zeros((len(dRR_in) - 1, 2), dtype=int)
            dRR[:, 0] = dRR_in[1:]
            dRR[:, 1] = dRR_in[:-1]

            point_sum = 0
            hist2d = np.zeros((300, 300))
            for i in range(len(dRR)):
                if abs(dRR[i, 0]) < 10 and abs(dRR[i, 1]) < 10:
                    continue

                x1 = dRR[i, 0] - bins + 150
                y1 = dRR[i, 1] - bins + 150
                x2 = dRR[i, 0] + bins + 150
                y2 = dRR[i, 1] + bins + 150
                x1 = check(x1, 300)
                y1 = check(y1, 300)
                x2 = check(x2, 300)
                y2 = check(y2, 300)

                point_sum = point_sum + 1
                hist2d[x1:x2, y1:y2] = 1
            if point_sum > 0:
                area_u = np.sum(hist2d)
                area_all = point_sum * (2 * bins) * (2 * bins)
                radius = area_u / area_all

    return radius


def check(d, max_d):
    d = int(np.round(d))
    if d < 0:
        d = 0
    elif d >= max_d:
        d = max_d
    return d


if __name__ == '__main__':
    from dataset import db_wfdb2020
    import matplotlib.pyplot as plt

    inputfiles = db_wfdb2020.get_filename_list()
    sampleL1 = []
    sampleL2 = []
    inputfiles = inputfiles[0:]
    for ii, ID in enumerate(inputfiles):
        if ii == 3:
            print('error')
        print('{}/{}'.format(ii, len(inputfiles)))
        rpos = db_wfdb2020.load_qrs(ID)
        label = db_wfdb2020.load_ecg_label_by_filename(ID, multi_label=True)
        RR = np.ediff1d(rpos) / 250
        sample_entropy = comput_Radius(RR)

        if 0 in label:
            sampleL1.append(sample_entropy)
        else:
            sampleL2.append(sample_entropy)

    sampleL1 = np.array(sampleL1).flatten()
    sampleL2 = np.array(sampleL2).flatten()
    data = [sampleL1, sampleL2]
    plt.boxplot(data)
    plt.legend(['af', 'other'])
    plt.show()

    from scipy import stats

    x1 = np.arange(len(sampleL1))
    x2 = np.arange(len(sampleL2))

    sampleL1 = np.nan_to_num(sampleL1, posinf=10)
    sampleL2 = np.nan_to_num(sampleL2, posinf=10)

    a = plt.hist(sampleL1, bins=100, density=True)
    b = plt.hist(sampleL2, bins=100, density=True)
    plt.plot(a[1][1:], a[0])
    plt.plot(b[1][1:], b[0])
    plt.legend(['af', 'other'])
    plt.show()
