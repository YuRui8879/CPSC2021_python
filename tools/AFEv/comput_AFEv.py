# % //This software is licensed under the BSD 3 Clause license: http://opensource.org/licenses/BSD-3-Clause
# %
# %
# % //Copyright (c) 2013, University of Oxford
# % //All rights reserved.
# %
# % //Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
# %
# % //Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
# % //Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
# % //Neither the name of the University of Oxford nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.
# % //THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# %   The method implemented in this file has been patented by their original
# %   authors. Commercial use of this code is thus strongly not
# %   recocomended.
# %
# % //Authors: 	Gari D Clifford -
# % //            Roberta Colloca -
# % //			Julien Oster	-
#
# Ported from Matlab code by mg.shao@qq.com 2020.6.17
#
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


def BPcount(sZ):
    # %bdc is the BIN diagonal count: number of non empty bins contained in
    # %the i-th diagonal of Z
    bdc = 0
    BC = 0
    # %pdc is the POINTS diagonal count: number of {dRR(i),dRR(i-1)} contained in
    # %the i-th diagonal of Z
    pdc = 0
    PC = 0
    #
    for i in range(-2, 3):
        bdc = np.sum(np.diag(sZ, i) != 0)
        pdc = np.sum(np.diag(sZ, i))
        BC = BC + bdc
        PC = PC + pdc
        sZ = sZ - np.diag(np.diag(sZ, i), i)

    return BC, PC, sZ


def metrics(dRR_in):
    # dRR=[dRR(2:length(dRR),1) dRR(1:length(dRR)-1,1)];
    dRR = np.zeros((len(dRR_in) - 1, 2))
    dRR[:, 0] = dRR_in[1:]
    dRR[:, 1] = dRR_in[:-1]

    # % COMPUTE OriginCount
    # OCmask=0.02;
    # os=sum(abs(dRR)<=OCmask,2);
    # OriginCount=sum(os==2);

    OCmask = 0.02
    os = np.sum(np.abs(dRR) <= OCmask, axis=1)
    OriginCount = np.sum(os == 2)

    #
    # % DELETE OUTLIERS |dRR|>=1.5
    OLmask = 1.5
    dRRnew = []
    for i in range(dRR.shape[0]):
        if np.sum(np.abs(dRR[i, :]) >= OLmask) == 0:
            dRRnew.append(dRR[i, :])

    if len(dRRnew) == 0:
        dRRnew = np.zeros((1, 2))
    else:
        dRRnew = np.array(dRRnew)

    #
    # % BUILD HISTOGRAM
    #
    # % Specify bin centers of the histogram
    # bin_c = -0.58:0.04: 0.58
    bin_c = np.arange(-0.6, 0.6 + 0.04, 0.04)
    # bin_c = np.arange(-0.9375, 0.9375 + 0.0625, 0.0625)
    #
    # % Three dimensional histogram of bivariate data
    # Z = hist3(dRRnew, {bin_c bin_c});
    Z, *_ = np.histogram2d(dRRnew[:, 0], dRRnew[:, 1], bins=bin_c)

    #
    # % COMPUTE POINT COUNT ZERO
    # %O1=sum(sum(Z(14,15:16)));
    # %O2=sum(sum(Z(15:16,14:17)));
    # %O3=sum(sum(Z(17,15:16)));
    # %PC0=O1+O2+O3;
    #
    # % Clear SegmentZero
    # Z(14,15:16)=0;
    # Z(15:16,14:17)=0;
    # Z(17,15:16)=0;

    Z[13, 14: 16] = 0
    Z[14: 16, 13: 17] = 0
    Z[16, 14: 16] = 0
    #
    # % [X,Y]=meshgrid(-0.58:0.04:0.58, -0.58:0.04:0.58);
    # % surf(X,Y, Z);
    # % axis tight
    # % xlabel('dRR(i-1)')
    # % ylabel('dRR(i)')
    #
    # %COMPUTE BinCount12
    # %COMPUTE PointCount12
    #
    # % Z2 contains all the bins belonging to the II quadrant of Z
    # Z2=Z(16:30,16:30);
    # [BC12,PC12,sZ2] = BPcount( Z2 );
    # Z(16:30,16:30)=sZ2;
    Z2 = Z[15:30, 15:30]
    BC12, PC12, sZ2 = BPcount(Z2)
    Z[15:30, 15:30] = sZ2

    #
    # %COMPUTE BinCount11
    # %COMPUTE PointCount11
    #
    # %Z3 cointains points belonging to the III quadrant of Z
    # Z3=Z(16:30,1:15);
    # Z3=fliplr(Z3);
    # [BC11,PC11,sZ3] = BPcount( Z3 );
    # Z(16:30,1:15)=fliplr(sZ3);
    Z3 = Z[15:30, 0:15]
    Z3 = np.fliplr(Z3)
    BC11, PC11, sZ3 = BPcount(Z3)
    Z[15:30, 0:15] = np.fliplr(sZ3)
    #
    # %COMPUTE BinCount10
    # %COMPUTE PointCount10
    #
    # %Z4 cointains points belonging to the IV quadrant of Z
    # Z4=Z(1:15,1:15);
    # [BC10,PC10,sZ4] = BPcount( Z4 );
    # Z(1:15,1:15)=sZ4;
    Z4 = Z[0:15, 0:15]
    BC10, PC10, sZ4 = BPcount(Z4)
    Z[0:15, 0:15] = sZ4

    #
    # %COMPUTE BinCount9
    # %COMPUTE PointCount9
    #
    # %Z1 cointains points belonging to the I quadrant of Z
    # Z1=Z(1:15,16:30);
    # Z1=fliplr(Z1);
    # [BC9,PC9,sZ1] = BPcount( Z1 );
    # Z(1:15,16:30)=fliplr(sZ1);
    Z1 = Z[0:15, 15:30]
    Z1 = np.fliplr(Z1)
    BC9, PC9, sZ1 = BPcount(Z1)
    Z[0:15, 15:30] = np.fliplr(sZ1)

    #
    # %COMPUTE BinCount5
    # BC5=sum(sum(Z(1:15,14:17)~=0));
    BC5 = np.sum(Z[0:15, 13:17] != 0)
    # %COMPUTE PointCount5
    # PC5=sum(sum(Z(1:15,14:17)));
    PC5 = np.sum(Z[0:15, 13:17])
    #
    # %COMPUTE BinCount7
    # BC7=sum(sum(Z(16:30,14:17)~=0));
    BC7 = np.sum(Z[15:30, 13:17] != 0)
    # %COMPUTE PointCount7
    # PC7=sum(sum(Z(16:30,14:17)));
    PC7 = np.sum(Z[15:30, 13:17])
    #
    # %COMPUTE BinCount6
    # BC6=sum(sum(Z(14:17,1:15)~=0));
    BC6 = np.sum(Z[13:17, 0:15] != 0)
    # %Compute PointCount6
    # PC6=sum(sum(Z(14:17,1:15)));
    PC6 = np.sum(Z[13:17, 0: 15])
    #
    # %COMPUTE BinCount8
    # BC8=sum(sum(Z(14:17,16:30)~=0));
    BC8 = np.sum(Z[13:17, 15:30] != 0)
    # %COMPUTE PointCount8
    # PC8=sum(sum(Z(14:17,16:30)));
    PC8 = np.sum(Z[13:17, 15: 30])
    #
    # % CLEAR SEGMENTS 5, 6, 7, 8
    #
    # % Clear segments 6 and 8
    # Z(14:17,:)=0;
    Z[13:17, :] = 0
    # %Clear segments 5 and 7
    # Z(:,14:17)=0;
    Z[:, 13:17] = 0
    #
    # % COMPUTE BinCount2
    # BC2=sum(sum(Z(1:13,1:13)~=0));
    BC2 = np.sum(Z[0:13, 0:13] != 0)
    # % COMPUTE PointCount2
    # PC2=sum(sum(Z(1:13,1:13)));
    PC2 = np.sum(Z[0:13, 0:13])
    #
    # % COMPUTE BinCount1
    # BC1=sum(sum(Z(1:13,18:30)~=0));
    BC1 = np.sum(Z[0:13, 17:30] != 0)
    # % COMPUTE PointCount1
    # PC1=sum(sum(Z(1:13,18:30)));
    PC1 = np.sum(Z[0:13, 17:30])
    #
    # % COMPUTE BinCount3
    # BC3=sum(sum(Z(18:30,1:13)~=0));
    BC3 = np.sum(Z[17:30, 0:13] != 0)
    # %COMPUTE PointCount3
    # PC3=sum(sum(Z(18:30,1:13)));
    PC3 = np.sum(Z[17:30, 0: 13])
    #
    # % COMPUTE BinCount4
    # BC4=sum(sum(Z(18:30,18:30)~=0));
    BC4 = np.sum(Z[18:30, 18:30] != 0)
    # % COMPUTE PointCount4
    # PC4=sum(sum(Z(18:30,18:30)));
    PC4 = np.sum(Z[17:30, 17:30])
    #
    #
    # % COMPUTE IrregularityEvidence
    IrrEv = BC1 + BC2 + BC3 + BC4 + BC5 + BC6 + BC7 + BC8 + BC9 + BC10 + BC11 + BC12

    # % COMPUTE PACEvidence
    PACEv = (PC1 - BC1) + (PC2 - BC2) + (PC3 - BC3) + (PC4 - BC4) + (PC5 - BC5) + (PC6 - BC6) + (PC10 - BC10) - (
            PC7 - BC7) - (PC8 - BC8) - (PC12 - BC12)

    # % COMPUTE AnisotropyEv
    AnisotropyEv = abs((PC9 + PC11) - (PC10 + PC12)) + abs((PC6 + PC7) - (PC5 + PC8))

    # % COMPUTE  DensityEv
    DensityEv = (PC5 - BC5) + (PC6 - BC6) + (PC7 - BC7) + (PC8 - BC8) + (PC9 - BC9) + (PC10 - BC10) + (PC11 - BC11) + (
            PC12 - BC12)

    # % COMPUTE RegularityEv
    # RegularityEv = 0

    return OriginCount, IrrEv, PACEv, AnisotropyEv, DensityEv


def comput_AFEv(RR_intervals):
    AFEv, IrrEv, OriginCount, PACEv, AnisotropyEv, DensityEv = 0, 0, 0, 0, 0, 0
    if len(RR_intervals) > 3:
        # %Compute dRR intervals series
        dRR = comp_dRR(RR_intervals)
        if len(dRR) > 3:
            # %Compute metrics
            OriginCount, IrrEv, PACEv, AnisotropyEv, DensityEv = metrics(dRR)

            # %Compute AFEvidence
            AFEv = IrrEv - OriginCount - 2 * PACEv

    return AFEv, IrrEv, OriginCount, PACEv, AnisotropyEv, DensityEv


if __name__ == '__main__':
    from dataset import db_wfdb2020
    import matplotlib.pyplot as plt

    inputfiles = db_wfdb2020.get_filename_list()
    sampleL1 = []
    sampleL2 = []
    inputfiles = inputfiles[0:]
    for ii, ID in enumerate(inputfiles):
        rpos = db_wfdb2020.load_qrs(ID)
        label = db_wfdb2020.load_ecg_label_by_filename(ID, multi_label=True)
        RR = np.ediff1d(rpos) / 250
        sample_entropy, *_ = comput_AFEv(RR)
        rr_len = len(RR) if len(RR) > 0 else 1
        sample_entropy = sample_entropy / rr_len
        if 0 in label:
            sampleL1.append(sample_entropy)
        else:
            sampleL2.append(sample_entropy)
        print('{}/{}'.format(ii, len(inputfiles)))
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
