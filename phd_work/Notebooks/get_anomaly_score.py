def get_anomaly_score(path_save, recos_id=None):
    reconstruction_data_pr = np.load(os.path.join(path_save, 'pr', 'reconstruction_data.npy'), allow_pickle=True)
    reconstruction_data_photon = np.load(os.path.join(path_save, 'photon', 'reconstruction_data.npy'), allow_pickle=True)

    num_det_pr = np.load(os.path.join(path_save, 'pr', 'num_det.npy'))
    num_det_photon = np.load(os.path.join(path_save, 'photon', 'num_det.npy'))

    dt_params_pr_signal = [d[:, 3:4] for d in dt_params_pr]
    dt_params_pr_time = [d[:, -1:] for d in dt_params_pr]
    dt_params_pr_signal_rubsov = [d[:, 3:4] for d in dt_params_rubsov_approximation_pr]
    dt_params_pr_time_rubsov = [d[:, -1:] for d in dt_params_rubsov_approximation_pr]
    dt_params_pr_signal_model = [d[:, 3:4] for d in reconstruction_data_pr]
    dt_params_pr_time_model = [d[:, 4:] for d in reconstruction_data_pr]

    dt_params_pr_sig_time = [np.concatenate([dt_params_pr_signal[i], dt_params_pr_time[i]], axis=1) for i in range(len(dt_params_pr_signal))]
    dt_params_pr_sig_time_rubsov = [np.concatenate([dt_params_pr_signal_rubsov[i], dt_params_pr_time_rubsov[i]], axis=1) for i in range(len(dt_params_pr_signal_rubsov))]
    dt_params_pr_sig_time_model = [np.concatenate([dt_params_pr_signal_model[i], dt_params_pr_time_model[i]], axis=1) for i in range(len(dt_params_pr_signal_model))]
    print(dt_params_pr_sig_time[0].shape, dt_params_pr_sig_time_rubsov[0].shape, dt_params_pr_sig_time_model[0].shape)

    dt_params_ph_signal = [d[:, 3:4] for d in dt_params_ph]
    dt_params_ph_time = [d[:, -1:] for d in dt_params_ph]
    dt_params_ph_signal_rubsov = [d[:, 3:4] for d in dt_params_rubsov_approximation_ph]
    dt_params_ph_time_rubsov = [d[:, -1:] for d in dt_params_rubsov_approximation_ph]
    dt_params_ph_signal_model = [d[:, 3:4] for d in reconstruction_data_photon]
    dt_params_ph_time_model = [d[:, 4:] for d in reconstruction_data_photon]

    dt_params_ph_sig_time = [np.concatenate([dt_params_ph_signal[i], dt_params_ph_time[i]], axis=1) for i in range(len(dt_params_ph_signal))]
    dt_params_ph_sig_time_rubsov = [np.concatenate([dt_params_ph_signal_rubsov[i], dt_params_ph_time_rubsov[i]], axis=1) for i in range(len(dt_params_ph_signal_rubsov))]
    dt_params_ph_sig_time_model = [np.concatenate([dt_params_ph_signal_model[i], dt_params_ph_time_model[i]], axis=1) for i in range(len(dt_params_ph_signal_model))]
    print(dt_params_ph_sig_time[0].shape, dt_params_ph_sig_time_rubsov[0].shape, dt_params_ph_sig_time_model[0].shape)

    embading_pr = np.load(os.path.join(path_save, 'pr', 'embading.npy'))
    embading_photon = np.load(os.path.join(path_save, 'photon', 'embading.npy'))
    recos_pr = np.load(os.path.join(path_save, 'pr', 'recos.npy'))
    recos_photon = np.load(os.path.join(path_save, 'photon', 'recos.npy'))

    recos_norming_pr = recos_pr * std_recos + mean_recos
    recos_norming_photon = recos_photon * std_recos + mean_recos

    from catboost import CatBoostRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import RobustScaler
    from sklearn.neighbors import KernelDensity

    ch_emb = 6
    num = 200000
    num_ph = 10000
    N = 13
    NUM_STEP = 3
    bandwidth = 0.3
    print(f'part of data {num_ph / num}')
    print('pr', embading_pr.shape, 'ph', embading_photon.shape)

    index = np.random.choice(len(embading_pr), num, replace=False)
    index_ph = np.random.choice(len(embading_photon), num_ph, replace=False)

    # features: latent + num_det
    feaches_pr = np.concatenate([embading_pr[index], num_det_pr[index].reshape(-1, 1)], axis=1)
    feaches_photon = np.concatenate([embading_photon[index_ph], num_det_photon[index_ph].reshape(-1, 1)], axis=1)
    feaches = np.concatenate([feaches_pr, feaches_photon], axis=0)

    label = np.concatenate([np.zeros(num), np.ones(num_ph)])
    recos_all_choise = np.concatenate([recos_pr[index], recos_photon[index_ph]], axis=0)

    shufle_index = np.random.permutation(len(feaches))
    feaches = feaches[shufle_index]
    label = label[shufle_index]
    recos_all_choise = recos_all_choise[shufle_index]
    num_det_feaches = feaches[:, ch_emb]

    # filter by num_det — keep feaches / recos / label aligned
    index_det = (N <= num_det_feaches) * (num_det_feaches < N + NUM_STEP)
    feaches = feaches[index_det]
    recos_all_choise = recos_all_choise[index_det]
    label_X = label[index_det]

    # latent only (drop num_det)
    X = feaches[:, :ch_emb].copy()
    print('X', X.shape, 'recos', recos_all_choise.shape)

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    print('pr', (label_X == 0).sum(), 'ph', (label_X == 1).sum())

    kde = KernelDensity(bandwidth=bandwidth).fit(X_scaled)
    scores_pr = kde.score_samples(X_scaled[label_X == 0])
    scores_ph = kde.score_samples(X_scaled[label_X == 1])
    print("CHECK", X_scaled.shape, label_X.shape, recos_all_choise.shape)
    recos_pr_choise = recos_all_choise[label_X == 0]
    recos_photon_choise = recos_all_choise[label_X == 1]
    print(scores_pr.mean(), scores_ph.mean())
    print(scores_pr.std(), scores_ph.std())

    fig_scores, ax_scores = plt.subplots()
    ax_scores.hist(scores_pr, bins=100, label='pr', histtype='step', density=True, log=True)
    ax_scores.hist(scores_ph, bins=100, label='ph', histtype='step', density=True, log=True)
    ax_scores.legend()
    ax_scores.set_title('KDE anomaly scores')

    # regression: latent -> recos (same number of rows)
    if recos_id is None:
        y = recos_all_choise
        loss_function = 'MultiRMSE'
    else:
        y = recos_all_choise[:, recos_id]
        loss_function = 'RMSE'

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model_regressor = CatBoostRegressor(
        n_estimators=1000,
        depth=4,
        learning_rate=0.1,
        random_state=42,
        loss_function=loss_function,
        verbose=False,
    )
    model_regressor.fit(X_train, y_train)
    predict = model_regressor.predict(X_test)
    mse = np.mean((predict - y_test) ** 2)
    print(f'mse: {mse}')

    fig_importance, ax_importance = plt.subplots()
    feat_names = [f'latent_{i}' for i in range(X.shape[1])]
    ax_importance.barh(feat_names, model_regressor.feature_importances_)
    ax_importance.set_title('CatBoostRegressor feature importances (latent -> recos)')
    print(model_regressor.feature_importances_)

    figs = {
        'feature_importances': fig_importance,
        'kde_scores': fig_scores,
    }

    return scores_pr, scores_ph, recos_pr_choise, recos_photon_choise, figs
