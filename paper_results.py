import numpy as np

if __name__ == '__main__':
    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_ETTh1_hl168_pl24_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN ETTh1 168 pred 24 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()

    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_ETTh1_hl168_pl72_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN ETTh1 168 pred 72 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()

    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_ETTh1_hl168_pl120_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN ETTh1 168 pred 120 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()


    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_WTH_hl168_pl24_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN WTH 168 pred 24 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()

    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_WTH_hl168_pl72_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN WTH 168 pred 72 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()

    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_WTH_hl168_pl120_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN WTH 168 pred 120 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()


    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_NASDAQ_hl60_pl15_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN WTH 60 pred 15 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()

    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_NASDAQ_hl60_pl30_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN WTH 60 pred 30 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()

    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_NASDAQ_hl60_pl60_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN WTH 60 pred 60 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()

    GMMFNN_maes = []
    GMMFNN_rmses = []
    for i in range(10):
        metric = np.load('./paper_results/GMM_FNN_NASDAQ_hl60_pl120_' + str(i) + '/metrics.npy', allow_pickle=True)
        GMMFNN_maes.append(metric[0])
        GMMFNN_rmses.append(metric[2])
    GMMFNN_maes = np.array(GMMFNN_maes)
    GMMFNN_rmses = np.array(GMMFNN_rmses)
    print("======================= GMM-FNN WTH 60 pred 120 ==========================")
    print("MAE: ", np.mean(GMMFNN_maes), '±', np.std(GMMFNN_maes))
    print("RMSE: ", np.mean(GMMFNN_rmses), '±', np.std(GMMFNN_rmses))
    print(GMMFNN_maes)
    print(GMMFNN_rmses)
    print()
    print()