import argparse
from pathlib import Path
import numpy as np
from scipy.stats import ttest_ind, wilcoxon, kendalltau
from sklearn.metrics import recall_score


def read_memfile(filename, shape, dtype='float32'):
    # read binary data and return as a numpy array
    fp = np.memmap(filename, dtype=dtype, mode='r', shape=shape)
    data = np.zeros(shape=shape, dtype=dtype)
    data[:] = fp[:]
    del fp


def scorePerformance(
        prMean_pred, prMean_true,
        rtMean_pred, rtMean_true,
        rrStd_pred, rrStd_true,
        ecgId_pred, ecgId_true):
    """
    Computes the combined multitask performance score. 
    The 3 regression tasks are individually scored using Kendalls 
    correlation coeffficient. 
    The user classification task is scored according to macro averaged 
    recall, with an adjustment for chance level. All performances are 
    clipped at 0.0, so that zero indicates chance or worse performance, 
    and 1.0 indicates perfect performance. 
    The individual performances are then combined by taking
    the geometric mean.
    :param prMean_pred: 1D float32 numpy array. 
        The predicted average P-R interval duration over the window. 
    :param prMean_true: 1D float32 numpy array. 
        The true average P-R interval duration over the window. 
    :param rtMean_pred: 1D float32 numpy array. 
        The predicted average R-T interval duration over the window.
    :param rtMean_true: 1D float32 numpy array. 
        The true average R-T interval duration over the window.
    :param rrStd_pred: 1D float32 numpy array. 
        The predicted R-R interval duration standard deviation over the window.
    :param rrStd_true: 1D float32 numpy array. 
        The true R-R interval duration standard deviation over the window. 
    :param ecgId_pred: 1D int32 numpy array. 
        The predicted user ID label for each window.
    :param ecgId_true: 1D int32 numpy array. 
        The true user ID label for each window.
    :return: 
        - The combined performance score on all tasks; 
                0.0 means at least one task has chance level performance 
                or worse while 1.0 means all tasks are solved perfectly.
        - The individual task performance scores are also returned
    """

    # Input checking
    assert isinstance(ecgId_pred, np.ndarray)
    assert len(ecgId_pred.shape) == 1
    assert ecgId_pred.dtype == np.int32

    assert isinstance(ecgId_true, np.ndarray)
    assert len(ecgId_true.shape) == 1
    assert ecgId_true.dtype == np.int32

    assert isinstance(rrStd_pred, np.ndarray)
    assert len(rrStd_pred.shape) == 1
    assert rrStd_pred.dtype == np.float32

    assert isinstance(rrStd_true, np.ndarray)
    assert len(rrStd_true.shape) == 1
    assert rrStd_true.dtype == np.float32

    assert isinstance(prMean_pred, np.ndarray)
    assert len(prMean_pred.shape) == 1
    assert prMean_pred.dtype == np.float32

    assert isinstance(prMean_true, np.ndarray)
    assert len(prMean_true.shape) == 1
    assert prMean_true.dtype == np.float32

    assert isinstance(rtMean_pred, np.ndarray)
    assert len(rtMean_pred.shape) == 1
    assert rtMean_pred.dtype == np.float32

    assert isinstance(rtMean_true, np.ndarray)
    assert len(rtMean_true.shape) == 1
    assert rtMean_true.dtype == np.float32

    assert (len(ecgId_pred) == len(ecgId_true)) \
        and (len(ecgId_pred) == len(prMean_pred)) \
        and (len(ecgId_pred) == len(prMean_true)) \
        and (len(ecgId_pred) == len(rtMean_pred)) \
        and (len(ecgId_pred) == len(rtMean_true)) \
        and (len(ecgId_pred) == len(rrStd_pred)) \
        and (len(ecgId_pred) == len(rrStd_true))

    # Accuracy is computed with macro averaged recall so that accuracy
    # is computed as though the classes were balanced, even if they are not.
    # Note that provided training, validation and testing sets are balanced.
    # Unbalanced classes would only be and issue if a new train/validation
    # split is created.
    # Any accuracy value worse than random chance will be clipped at zero.

    ecgIdAccuracy = recall_score(ecgId_true, ecgId_pred, average='macro')
    adjustementTerm = 1.0 / len(np.unique(ecgId_true))
    ecgIdAccuracy = (ecgIdAccuracy - adjustementTerm) / (1 - adjustementTerm)
    if ecgIdAccuracy < 0:
        ecgIdAccuracy = 0.0

    # Compute Kendall correlation coefficients for regression tasks.
    # Any coefficients worse than chance will be clipped to zero.
    rrStdTau, _ = kendalltau(rrStd_pred, rrStd_true)
    if rrStdTau < 0:
        rrStdTau = 0.0

    prMeanTau, _ = kendalltau(prMean_pred, prMean_true)
    if prMeanTau < 0:
        prMeanTau = 0.0

    rtMeanTau, _ = kendalltau(rtMean_pred, rtMean_true)
    if rtMeanTau < 0:
        rtMeanTau = 0.0

    # Compute the final performance score as the geometric mean of the
    # individual task performances. A high geometric mean ensures that
    # there are no tasks with very poor performance that are masked by good
    # performance on the other tasks. If any task has chance performance
    # or worse, the overall performance will be zero. If all tasks are
    # perfectly solved, the overall performance will be 1.

    combinedPerformanceScore = np.power(
        rrStdTau * prMeanTau * rtMeanTau * ecgIdAccuracy, 0.25)

    return (
        combinedPerformanceScore,
        prMeanTau, rtMeanTau, rrStdTau, ecgIdAccuracy
    )


def generate_data_for_kendaltau(pred, gt, k):
    data = []
    for i in range(gt.shape[0]):
        for j in range(i + 1, gt.shape[0]):
            v = None
            if (
                ((pred[i, k] > pred[j, k]) and (gt[i, k] > gt[j, k])) or
                ((pred[i, k] < pred[j, k]) and (gt[i, k] < gt[j, k]))
            ):
                v = 1.0
            elif (
                ((pred[i, k] > pred[j, k]) and (gt[i, k] < gt[j, k])) or
                ((pred[i, k] < pred[j, k]) and (gt[i, k] > gt[j, k]))
            ):
                v = -1.0
            else:
                v = 0.0
            data.append(v)

    return np.array(data)


def evaluate_stats(gt, pred1, pred2, alpha=0.05):

    assert len(gt) == len(pred1) == len(
        pred2), "Amount of samples is not identical"
    assert gt.shape[1] == pred1.shape[1] == pred2.shape[
        1] == 4, "Amount of metrics is not identical"

    perf1 = scorePerformance(
        pred1[:, 0], gt[:, 0],
        pred1[:, 1], gt[:, 1],
        pred1[:, 2], gt[:, 2],
        pred1[:, 3].astype(np.int32), gt[:, 3].astype(np.int32)
    )

    perf2 = scorePerformance(
        pred2[:, 0], gt[:, 0],
        pred2[:, 1], gt[:, 1],
        pred2[:, 2], gt[:, 2],
        pred2[:, 3].astype(np.int32), gt[:, 3].astype(np.int32)
    )

    # data for running stat tests

    pr_pred1 = generate_data_for_kendaltau(pred1, gt, 0)
    pr_pred2 = generate_data_for_kendaltau(pred2, gt, 0)

    rt_pred1 = generate_data_for_kendaltau(pred1, gt, 1)
    rt_pred2 = generate_data_for_kendaltau(pred2, gt, 1)

    rr_pred1 = generate_data_for_kendaltau(pred1, gt, 2)
    rr_pred2 = generate_data_for_kendaltau(pred2, gt, 2)

    acc_pred1 = (pred1[:, 3].astype(np.int32) == gt[
                 :, 3].astype(np.int32)).astype(np.int32)
    acc_pred2 = (pred2[:, 3].astype(np.int32) == gt[
                 :, 3].astype(np.int32)).astype(np.int32)

    acc_t, acc_pvalue = wilcoxon(acc_pred1, acc_pred2, zero_method='zsplit')
    pr_t, pr_pvalue = wilcoxon(pr_pred1, pr_pred2, zero_method='zsplit')
    rt_t, rt_pvalue = wilcoxon(rt_pred1, rt_pred2, zero_method='zsplit')
    rr_t, rr_pvalue = wilcoxon(rr_pred1, rr_pred2, zero_method='zsplit')

    output = []

    if pr_pvalue >= alpha:
        # Can't reject the null hypothesis
        # ==> No Significant difference between the two
        print(
            "\nPR_Mean: p-value {} : No difference between the two models ".format(pr_pvalue))
        output.append(0)
    else:
        # we reject the null hypothesis
        print(
            "\nPR_Mean: p-value {} : Difference between the two models ".format(pr_pvalue))
        a, b = 'model1', 'model2'
        if perf1[1] < perf2[1]:
            a, b = b, a
            output.append(-1)
        else:
            output.append(1)
        print("PR_Mean: {} is better than {}.".format(a, b))

    if rt_pvalue >= alpha:
        # Can't reject the null hypothesis
        # ==> No Significant difference between the two
        print(
            "\nRT_Mean: p-value {} : No difference between the two models ".format(rt_pvalue))
        output.append(0)
    else:
        # we reject the null hypothesis
        print(
            "\nRT_Mean: p-value {} : Difference between the two models ".format(rt_pvalue))
        a, b = 'model1', 'model2'
        if perf1[2] < perf2[2]:
            a, b = b, a
            output.append(-1)
        else:
            output.append(1)
        print("RT_Mean: {} is better than {}.".format(a, b))

    if rr_pvalue >= alpha:
        # Can't reject the null hypothesis
        # ==> No Significant difference between the two
        print(
            "\nRR_Stdev: p-value {} : No difference between the two models ".format(rr_pvalue))
        output.append(0)
    else:
        # we reject the null hypothesis
        print(
            "\nRR_Stdev: p-value {} : Difference between the two models ".format(rr_pvalue))
        a, b = 'model1', 'model2'
        if perf1[3] < perf2[3]:
            a, b = b, a
            output.append(-1)
        else:
            output.append(1)
        print("RR_Stdev: {} is better than {}.".format(a, b))

    if acc_pvalue >= alpha:
        # Can't reject the null hypothesis
        # ==> No Significant difference between the two
        print(
            "\nAccuracy: p-value {} : No difference between the two models ".format(acc_pvalue))
        output.append(0)
    else:
        # we reject the null hypothesis
        print(
            "\nAccuracy: p-value {} : Difference between the two models ".format(acc_pvalue))
        a, b = 'model1', 'model2'
        if perf1[4] < perf2[4]:
            a, b = b, a
            output.append(-1)
        else:
            output.append(1)
        print("Accuracy: {} is better than {}.".format(a, b))

    perf = [perf1, perf2]
    stats = [[acc_t, acc_pvalue], [pr_t, pr_pvalue],
             [rt_t, rt_pvalue], [rr_t, rr_pvalue]]

    return output, perf, stats


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--gt", type=str, required=True,
                        help='path to the ground truth data')
    parser.add_argument("--pred1", type=str, required=True,
                        help='path to the 1st prediction data')
    parser.add_argument("--pred2", type=str, required=True,
                        help='path to the 2nd prediction data')
    parser.add_argument("--numsample", type=int, default=160,
                        help='num of sample in the dateset')
    parser.add_argument("--alpha", type=float, default=0.05,
                        help='alpha threshold for p-value comparison')

    args = parser.parse_args()
    numsample = args.numsample

    gt = read_memfile(args.gt, (numsample, 4))
    pred1 = read_memfile(args.pred1, (numsample, 4))
    pred2 = read_memfile(args.pred2, (numsample, 4))

    evaluate_stats(gt, pred1, pred2, args.alpha)


if __name__ == "__main__":

    main()
