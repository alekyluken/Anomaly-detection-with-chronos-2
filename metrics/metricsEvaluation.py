import numpy as np
import copy

from sklearn import metrics

def get_metrics(score, labels, slidingWindow=100, pred=None, thre=250):
    """
    Compute various evaluation metrics for anomaly detection.
    
    Args:
        score (np.ndarray): Anomaly scores.
        labels (np.ndarray): Ground truth labels.
        slidingWindow (int): Size of the sliding window for range-based metrics.
        pred (np.ndarray or None): Predicted binary labels. If None, oracle threshold is used.
        thre (int): Threshold parameter for range AUC calculation.
        
    Returns:
        dict: A dictionary containing various evaluation metrics.
    """
    grader = basic_metricor()

    VUS_ROC, VUS_PR = grader.RangeAUC_volume_opt_mem(labels_original=labels.astype(int), score=score, windowSize=slidingWindow, thre=thre)


    '''
    Threshold Dependent
    if pred is None --> use the oracle threshold
    '''

    return {
        'AUC-PR': grader.metric_PR(labels, score),
        'AUC-ROC': grader.metric_ROC(labels, score),
        'VUS-PR': VUS_PR,
        'VUS-ROC': VUS_ROC,

        'Standard-F1': grader.metric_PointF1(labels, score, preds=pred),
        'PA-F1': grader.metric_PointF1PA(labels, score, preds=pred),
        'Event-based-F1': grader.metric_EventF1PA(labels, score, preds=pred),
        'R-based-F1': grader.metric_RF1(labels, score, preds=pred)
    }


class basic_metricor():
    def __init__(self, a = 1, probability = True, bias = 'flat', ):
        self.a = a
        self.probability = probability
        self.bias = bias
        self.eps = 1e-15

    def w(self, AnomalyRange, p):
        MyValue = 0
        MaxValue = 0
        start = AnomalyRange[0]
        AnomalyLength = AnomalyRange[1] - AnomalyRange[0] + 1
        for i in range(start, start +AnomalyLength):
            bi = self.b(i, AnomalyLength)
            MaxValue +=  bi
            if i in p:
                MyValue += bi
        return MyValue/MaxValue

    def Cardinality_factor(self, Anomolyrange, Prange):
        score = 0
        start = Anomolyrange[0]
        end = Anomolyrange[1]

        for i in Prange:
            if i[0] >= start and i[0] <= end:
                score +=1
            elif start >= i[0] and start <= i[1]:
                score += 1
            elif end >= i[0] and end <= i[1]:
                score += 1
            elif start >= i[0] and end <= i[1]:
                score += 1

        return 0 if score == 0 else (1/score)

    def b(self, i, length):
        match self.bias:
            case 'flat': return 1
            case 'front-end bias': return length - i + 1
            case 'back-end bias': return i
            case _: return i if i <= length/2 else (length - i + 1)

    def _adjust_predicts(self, score, label, threshold=None, pred=None):
        """
        Calculate adjusted predict labels using given `score`, `threshold` (or given `pred`) and `label`.

        Args:
            score (np.ndarray): The anomaly score
            label (np.ndarray): The ground-truth label
            threshold (float): The threshold of anomaly score.
                A point is labeled as "anomaly" if its score is higher than the threshold.
            pred (np.ndarray or None): if not None, adjust `pred` and ignore `score` and `threshold`,

        Returns:
            np.ndarray: predict labels
        """
        if len(score) != len(label):
            raise ValueError("score and label must have the same length")
        score = np.asarray(score)

        predict = (score > threshold) if pred is None else copy.deepcopy(pred)
        actual = np.asarray(label) > 0.1
        anomaly_state = False
        anomaly_count = 0
        for i in range(len(score)):
            if actual[i] and predict[i] and not anomaly_state:
                    anomaly_state = True
                    anomaly_count += 1
                    for j in range(i, 0, -1):
                        if not actual[j]:
                            break
                        else:
                            if not predict[j]:
                                predict[j] = True
            elif not actual[i]:
                anomaly_state = False
            if anomaly_state:
                predict[i] = True
        return predict

    def metric_ROC(self, label, score):
        return metrics.roc_auc_score(label, score)

    def metric_PR(self, label, score):
        return metrics.average_precision_score(label, score)

    def metric_PointF1(self, label, score, preds=None):
        if preds is None:
            precision, recall, _ = metrics.precision_recall_curve(label, score)
            return np.max(2 * (precision * recall) / (precision + recall + 0.00001))
        else:
            return metrics.precision_recall_fscore_support(label, preds, zero_division=0)[2][1]


    def metric_RF1(self, label, score, preds=None):
        if preds is None:
            Rf1_scores = []

            for threshold in np.linspace(score.min(), score.max(), 100):
                preds = (score > threshold).astype(int)

                Rrecall = self.range_recall_new(label, preds, alpha=0.2)
                Rprecision = self.range_recall_new(preds, label, 0)

                Rf1_scores.append(0 if Rprecision + Rrecall == 0 else (2 * Rrecall * Rprecision / (Rprecision + Rrecall)))

            return max(Rf1_scores)
        else:
            Rrecall = self.range_recall_new(label, preds, alpha=0.2)
            Rprecision = self.range_recall_new(preds, label, 0)
            return 0 if Rprecision + Rrecall==0 else (2 * Rrecall * Rprecision / (Rprecision + Rrecall))


    def metric_PointF1PA(self, label, score, preds=None):
        if preds is None:
            return max([metrics.f1_score(label, self._adjust_predicts(score, label, pred=(score > threshold).astype(int)))
                for threshold in np.linspace(score.min(), score.max(), 100)])
        else:
            return metrics.f1_score(label, self._adjust_predicts(score, label, pred=preds))


    def _get_events(self, y_test, outlier=1, normal=0):
        events = dict()
        label_prev = normal
        event = 0  # corresponds to no event
        event_start = 0
        for tim, label in enumerate(y_test):
            if label == outlier:
                if label_prev == normal:
                    event += 1
                    event_start = tim
            else:
                if label_prev == outlier:
                    event_end = tim - 1
                    events[event] = (event_start, event_end)
            label_prev = label

        if label_prev == outlier:
            event_end = tim - 1
            events[event] = (event_start, event_end)
        return events

    def metric_EventF1PA(self, label, score, preds=None):
        true_events = self._get_events(label)

        if preds is None:
            thresholds = np.linspace(score.min(), score.max(), 100)
            EventF1PA_scores = []
            for threshold in thresholds:
                preds = (score > threshold).astype(int)

                tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
                rec_e = tp/(tp + len(true_events) - tp)
                prec_t = metrics.precision_score(label, preds)
                EventF1PA = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

                EventF1PA_scores.append(EventF1PA)

            EventF1PA1 = max(EventF1PA_scores)
        else:
            tp = np.sum([preds[start:end + 1].any() for start, end in true_events.values()])
            rec_e = tp/(tp + len(true_events) - tp) 
            prec_t = metrics.precision_score(label, preds)
            EventF1PA1 = 2 * rec_e * prec_t / (rec_e + prec_t + self.eps)

        return EventF1PA1

    def range_recall_new(self, labels, preds, alpha):
        p = np.where(preds == 1)[0]    # positions of predicted label==1
        range_pred = self.range_convers_new(preds)
        range_label = self.range_convers_new(labels)

        score = alpha * self.existence_reward(range_label, preds) + \
            (1-alpha) * np.sum(self.w(i, p) * self.Cardinality_factor(i, range_pred) for i in range_label)
        
        return (score/len(range_label)) if len(range_label) != 0 else 0
    
        
    def range_convers_new(self, label):
        '''
        input: arrays of binary values
        output: list of ordered pair [[a0,b0], [a1,b1]... ] of the inputs
        '''
        anomaly_starts = np.where(np.diff(label) == 1)[0] + 1
        anomaly_ends, = np.where(np.diff(label) == -1)
        if len(anomaly_ends):
            if not len(anomaly_starts) or anomaly_ends[0] < anomaly_starts[0]:
                # we started with an anomaly, so the start of the first anomaly is the start of the labels
                anomaly_starts = np.concatenate([[0], anomaly_starts])
        if len(anomaly_starts):
            if not len(anomaly_ends) or anomaly_ends[-1] < anomaly_starts[-1]:
                # we ended on an anomaly, so the end of the last anomaly is the end of the labels
                anomaly_ends = np.concatenate([anomaly_ends, [len(label) - 1]])
        return list(zip(anomaly_starts, anomaly_ends))


    def existence_reward(self, labels, preds):
        '''
        labels: list of ordered pair
        preds predicted data
        '''
        return sum([1 for i in labels if preds[i[0]:i[1]+1].any()])
        
    def new_sequence(self, label, sequence_original, window):
        a = max(sequence_original[0][0] - window // 2, 0)
        sequence_new = []
        for i in range(len(sequence_original) - 1):
            if sequence_original[i][1] + window // 2 < sequence_original[i + 1][0] - window // 2:
                sequence_new.append((a, sequence_original[i][1] + window // 2))
                a = sequence_original[i + 1][0] - window // 2
        sequence_new.append((a, min(sequence_original[len(sequence_original) - 1][1] + window // 2, len(label) - 1)))
        return sequence_new

    def sequencing(self, x, L, window=5):
        label = x.copy().astype(float)

        for k in range(len(L)):
            s = L[k][0]
            e = L[k][1]

            x1 = np.arange(e + 1, min(e + window // 2 + 1, len(label)))
            label[x1] += np.sqrt(1 - (x1 - e) / (window))

            x2 = np.arange(max(s - window // 2, 0), s)
            label[x2] += np.sqrt(1 - (s - x2) / (window))

        return np.minimum(np.ones(len(label)), label)

    def RangeAUC_volume_opt_mem(self, labels_original, score, windowSize, thre=250):
        window_3d = np.arange(0, windowSize + 1, 1)
        P = np.sum(labels_original)
        seq = self.range_convers_new(labels_original)
        l = self.new_sequence(labels_original, seq, windowSize)

        score_sorted = -np.sort(-score)

        tpr_3d = np.zeros((windowSize + 1, thre + 2))
        fpr_3d = np.zeros((windowSize + 1, thre + 2))
        prec_3d = np.zeros((windowSize + 1, thre + 1))

        auc_3d = np.zeros(windowSize + 1)
        ap_3d = np.zeros(windowSize + 1)

        tp = np.zeros(thre)
        N_pred = np.zeros(thre)
        p = np.zeros((thre, len(score)))

        for k, i in enumerate(np.linspace(0, len(score) - 1, thre).astype(int)):
            pred = score >= score_sorted[i]
            p[k] = pred
            N_pred[k] = np.sum(pred)

        for window in window_3d:
            labels_extended = self.sequencing(labels_original, seq, window)
            L = self.new_sequence(labels_extended, seq, window)

            TF_list = np.zeros((thre + 2, 2))
            Precision_list = np.ones(thre + 1)
            j = 0

            for i in np.linspace(0, len(score) - 1, thre).astype(int):
                labels = labels_extended.copy()
                existence = 0

                for seg in L:
                    labels[seg[0]:seg[1] + 1] = labels_extended[seg[0]:seg[1] + 1] * p[j][seg[0]:seg[1] + 1]
                    if (p[j][seg[0]:(seg[1] + 1)] > 0).any():
                        existence += 1
                for seg in seq:
                    labels[seg[0]:seg[1] + 1] = 1

                N_labels = 0
                TP = 0
                for seg in l:
                    TP += np.dot(labels[seg[0]:seg[1] + 1], p[j][seg[0]:seg[1] + 1])
                    N_labels += np.sum(labels[seg[0]:seg[1] + 1])

                TP += tp[j]
                FP = N_pred[j] - TP

                existence_ratio = existence / len(L)

                P_new = (P + N_labels) / 2
                recall = min(TP / P_new, 1)

                TPR = recall * existence_ratio

                N_new = len(labels) - P_new
                FPR = FP / N_new
                Precision = TP / N_pred[j]
                j += 1

                TF_list[j] = [TPR, FPR]
                Precision_list[j] = Precision

            TF_list[j + 1] = [1, 1]
            tpr_3d[window] = TF_list[:, 0]
            fpr_3d[window] = TF_list[:, 1]
            prec_3d[window] = Precision_list

            auc_3d[window] = (np.dot(TF_list[1:, 1] - TF_list[:-1, 1], (TF_list[1:, 0] + TF_list[:-1, 0]) / 2))
            ap_3d[window] = (np.dot(TF_list[1:-1, 0] - TF_list[:-2, 0], Precision_list[1:]))
        return sum(auc_3d) / len(window_3d), sum(ap_3d) / len(window_3d)