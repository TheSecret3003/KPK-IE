from get_candidates import get_candidates
from evaluation import clean_labeled_data, get_true_references, predict_similar_entities, calc_accuracy_pct
import pandas as pd

if __name__ == "__main__":
    # print("kai: ", get_candidates("kai"))
    # print("pelni: ", get_candidates("pelni"))
    # print("banyumas: ", get_candidates("banyumas"))
    # print("PT Telkom Indonesia: ", get_candidates("PT Telkom Indonesia".lower()))
    # print("Kementerian BUMN: ", get_candidates("Kementerian BUMN".lower()))
    """
    Sample output:
    kai:  {'pt kai': 3, 'pt kereta api indonesia persero kai': 32}
    pelni:  {'pt pelni': 3, 'pt pelayaran nasional indonesia persero pelni': 40}
    banyumas:  {'pemkab banyumas': 7, 'pemerintah banyumas': 11, 'pemerintah kabupaten banyumas': 21}
    """

    # df_tr = pd.read_csv('../indexing/data/splitted_data/v1/train.csv')
    # df_val = pd.read_csv('../indexing/data/splitted_data/v1/val.csv')
    # df_test = pd.read_csv('../indexing/data/splitted_data/v1/test.csv')
    # df_all = pd.concat([df_tr, df_val, df_test])
    # df_cln = clean_labeled_data(df_all)
    # # print(df_cln.head())

    # true_ref = get_true_references(df_cln, 'mahkamah agung')
    # # print(len(true_ref))
    # # print(true_ref)

    # # prep test data
    # test_df = pd.read_csv('../indexing/data/indexing_v1/reference_data.csv')[:100]
    # preds = predict_similar_entities(test_df, df_cln)
    # preds.to_csv('./eval/performance_test.csv')
    # print(preds)

    # preds = pd.read_csv('./eval/performance_test.csv')
    # preds_ = calc_accuracy_pct(preds)
    # preds_.to_csv('./eval/performance_test_acc.csv')
    # print(preds_)

    