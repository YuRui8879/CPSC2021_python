import numpy as np
from sklearn.linear_model import LinearRegression, RANSACRegressor, HuberRegressor, TheilSenRegressor


def find_peak(ecg):
    diff_ecg = np.ediff1d(ecg)
    pos1 = np.argmax(diff_ecg)
    pos2 = np.argmin(diff_ecg)
    if pos1 < pos2:
        start = pos1
        end = pos2 + 1
        r_peak = start + np.argmax(ecg[start:end])
    else:
        start = pos2
        end = pos1 + 1
        r_peak = start + np.argmin(ecg[start:end])

    return r_peak


def compute_R_peak(ecg_mask, ecg, min_width=(24 + 24 // 2), shrink=8):
    pos_list = []
    ecg_mask[0] = 0
    ecg_mask[-1] = 0

    diff_ecg = np.ediff1d(ecg)

    p_start = ecg_mask[1:] - ecg_mask[:-1]
    p_end = ecg_mask[:-1] - ecg_mask[1:]
    start_pos = np.argwhere(p_start == 1)[:, 0]
    end_pos = np.argwhere(p_end == 1)[:, 0]
    for ii in range(len(start_pos)):
        width = end_pos[ii] - start_pos[ii] + 1
        if width >= min_width:
            r_peak = np.argmax(np.abs(diff_ecg[start_pos[ii] + shrink:end_pos[ii] + 1 - shrink]))
            pos_list.append(start_pos[ii] + shrink + r_peak)

    return np.array(pos_list, dtype=np.int32)


def compute_RR_interval(R_peak):
    RR = np.ediff1d(R_peak)
    return RR


def compute_mean_RR(rr):
    if len(rr) > 5:
        mean_rr = np.median(rr)
        rr = rr[rr < mean_rr * 2]
        if len(rr) > 5:
            mean_rr = np.median(rr)
        elif len(rr) > 0:
            mean_rr = np.mean(rr)
        else:
            mean_rr = 250
    elif len(rr) > 0:
        mean_rr = np.mean(rr)
    else:
        mean_rr = 250

    return mean_rr


# estimators = [('OLS', LinearRegression()),
#               ('Theil-Sen', TheilSenRegressor(random_state=42)),
#               ('RANSAC', RANSACRegressor(random_state=42)),
#               ('Huber', HuberRegressor())]

def compute_mean_RR_robust(RRs, debug=0, estimator='Huber', fs=250):
    np.seterr(divide='ignore', invalid='ignore', over='ignore')
    if len(RRs) <= 5:
        rr_mean = compute_mean_RR(RRs)
    else:
        try:
            X_train = np.array(range(len(RRs))).reshape(-1, 1)
            y_train = RRs
            X_test = np.array([len(RRs) / 2 - 1]).reshape(1, -1)
            # X_test = X_train

            # RANSAC回归

            if estimator == 'RANSAC':
                ran_model = RANSACRegressor(random_state=0)
            elif estimator == 'Huber':
                ran_model = HuberRegressor()
            elif estimator == 'Theil-Sen':
                ran_model = TheilSenRegressor()
            elif estimator == 'OLS':
                ran_model = LinearRegression()

            ran_model.fit(X_train, y_train)

            ran_y_pred = ran_model.predict(X_test)

            if len(ran_y_pred.shape) > 1:
                rr_mean = ran_y_pred[0, 0]
            else:
                rr_mean = ran_y_pred[0]

            if debug:
                print(estimator + ":" + str(rr_mean))
                import matplotlib.pyplot as plt
                # 评价模型的效果
                plt.figure(figsize=(15, 10))

                plt.scatter(X_test, ran_y_pred, c='red', label='Outliers')
                plt.plot(X_train, y_train, '-k', )
                plt.legend(loc='upper right')
                plt.show()
        except Exception as e:
            print('compute_mean_RR_robust', e)
            rr_mean = compute_mean_RR(RRs)

    if rr_mean == 0:
        rr_mean = fs
    return rr_mean


if __name__ == '__main__':
    import tensorflow as tf
    from dataset import db_wfdb2020

    with tf.device('/cpu:0'):
        inputfiles = db_wfdb2020.get_filename_list()
        # inputfiles = inputfiles[489:490]
        for ID in inputfiles:
            print(ID, end=' ')
            rpos = db_wfdb2020.load_qrs(ID)
            RRs = compute_RR_interval(rpos)
            rr_mean = compute_mean_RR_robust(RRs)
            print(rr_mean)
