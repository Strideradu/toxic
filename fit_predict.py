from toxic.model import *
from toxic.nltk_utils import tokenize_sentences
from toxic.train_utils import train_folds
from toxic.embedding_utils import read_embedding_list, clear_embedding_list, convert_tokens_to_ids

import argparse
import numpy as np
import os
import pandas as pd

UNKNOWN_WORD = "_UNK_"
END_WORD = "_END_"
NAN_WORD = "_NAN_"

CLASSES = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

PROBABILITIES_NORMALIZE_COEFFICIENT = 1.4


def main():
    parser = argparse.ArgumentParser(
        description="Recurrent neural network for identifying and classifying toxic online comments")

    parser.add_argument("train_file_path")
    parser.add_argument("test_file_path")
    parser.add_argument("embedding_path")
    parser.add_argument("--result-path", default="toxic_results")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--sentences-length", type=int, default=500)
    parser.add_argument("--recurrent-units", type=int, default=64)
    parser.add_argument("--dropout-rate", type=float, default=0.3)
    parser.add_argument("--dense-size", type=int, default=32)
    parser.add_argument("--fold-count", type=int, default=10)
    parser.add_argument("--aug-count", type=int, default=1)

    args = parser.parse_args()

    if args.fold_count <= 1:
        raise ValueError("fold-count should be more than 1")

    print("Loading data...")
    train_data = pd.read_csv(args.train_file_path)
    test_data = pd.read_csv(args.test_file_path)

    embed_path = os.path.join(args.embedding_path, 'embeddings.npz')
    data = np.load(embed_path)
    embedding_matrix = data['arr_0']

    data_path = os.path.join(args.embedding_path, 'train.npz')
    data = np.load(data_path)
    X_train = data['arr_0']

    data_path = os.path.join(args.embedding_path, 'test.npz')
    data = np.load(data_path)
    X_test = data['arr_0']

    data_path = os.path.join(args.embedding_path, 'label.npz')
    data = np.load(data_path)
    y_train = data['arr_0']


    get_model_func = lambda: get_LSTMGRU_GlobalMaxAve(
        embedding_matrix,
        args.sentences_length,
        args.dropout_rate,
        args.recurrent_units,
        args.dense_size)

    print("Starting to train models...")
    models, train_preds = train_folds(X_train, y_train, args.fold_count, args.batch_size, get_model_func, aug=args.aug_count)

    if not os.path.exists(args.result_path):
        os.mkdir(args.result_path)

    print("Predicting results...")
    test_predicts_list = []
    for fold_id, model in enumerate(models):
        model_path = os.path.join(args.result_path, "model{0}_weights.npy".format(fold_id))
        np.save(model_path, model.get_weights())

        test_predicts_path = os.path.join(args.result_path, "test_predicts{0}.npy".format(fold_id))
        test_predicts = model.predict(X_test, batch_size=args.batch_size)
        test_predicts_list.append(test_predicts)
        np.save(test_predicts_path, test_predicts)

    test_predicts = np.ones(test_predicts_list[0].shape)
    for fold_predict in test_predicts_list:
        test_predicts *= fold_predict

    test_predicts **= (1. / len(test_predicts_list))
    test_predicts **= PROBABILITIES_NORMALIZE_COEFFICIENT

    test_ids = test_data["id"].values
    test_ids = test_ids.reshape((len(test_ids), 1))

    test_predicts = pd.DataFrame(data=test_predicts, columns=CLASSES)
    test_predicts["id"] = test_ids
    test_predicts = test_predicts[["id"] + CLASSES]
    submit_path = os.path.join(args.result_path, "submit")
    test_predicts.to_csv(submit_path, index=False)

    train_ids =train_data["id"].values
    train_ids = train_ids.reshape((len(train_ids), 1))

    train_predicts = pd.DataFrame(data=train_preds, columns=CLASSES)
    train_predicts["id"] = train_ids
    train_predicts = train_predicts[["id"] + CLASSES]
    valid_path = os.path.join(args.result_path, "valid")
    train_predicts.to_csv(valid_path, index=False)


if __name__ == "__main__":
    main()
