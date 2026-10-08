# bin by np.log10(E), E = rec * std + mean + 18.5
def bin_by_log10(rec, n_bins=100):
    rec_pr_norm = rec * std_recos[3] + mean_recos[3] + 18.5
    log_rec = np.log10(rec_pr_norm)
    bins = np.linspace(log_rec.min(), log_rec.max(), n_bins)
    indexes = []
    mean_bins = []
    for i in range(len(bins) - 1):
        index = np.where((log_rec >= bins[i]) & (log_rec < bins[i + 1]))[0]
        indexes.append(index)
        mean_bins.append((bins[i] + bins[i + 1]) / 2)
    return indexes, mean_bins


def plot_anomaly_score(anomaly_score, rec):
    indexes, mean_bins = bin_by_log10(rec)
    mean_anomaly_score = []
    std_anomaly_score = []
    mean_bins_used = []

    for index, mean_bin in zip(indexes, mean_bins):
        if len(index) == 0:
            continue
        scores_bin = anomaly_score[index]
        mean_bins_used.append(mean_bin)
        mean_anomaly_score.append(scores_bin.mean())
        std_anomaly_score.append(scores_bin.std())

    return mean_bins_used, mean_anomaly_score, std_anomaly_score


mean_recos_1, mean_anomaly_score_1, std_anomaly_score_1 = plot_anomaly_score(scores_pr_1, rec_pr)
mean_recos_2, mean_anomaly_score_2, std_anomaly_score_2 = plot_anomaly_score(scores_pr_2, rec_pr_2)
