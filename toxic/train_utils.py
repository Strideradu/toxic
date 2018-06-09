from sklearn.metrics import log_loss
from sklearn.metrics import roc_auc_score

import numpy as np


def _train_model(model, batch_size, train_x, train_y, val_x, val_y):
    best_loss = -1
    best_weights = None
    best_epoch = 0
    best_auc = 0
    best_pred = None

    current_epoch = 0

    while True:
        model.fit(train_x, train_y, batch_size=batch_size, epochs=1)
        y_pred = model.predict(val_x, batch_size=batch_size)
        auc = roc_auc_score(val_y, y_pred)

        total_loss = 0
        for j in range(6):
            loss = log_loss(val_y[:, j], y_pred[:, j])
            total_loss += loss

        total_loss /= 6.



        #print("Epoch {0} loss {1} best_loss {2} corresponding auc {3}".format(current_epoch, total_loss, best_loss, best_auc))
        print("Epoch {0} auc {1} total_loss {2} best auc {3}".format(current_epoch, auc, total_loss,
                                                                              best_auc))

        current_epoch += 1
        """
        if total_loss < best_loss or best_loss == -1:
            best_loss = total_loss
            best_auc = auc
            best_weights = model.get_weights()
            best_epoch = current_epoch
        else:
            if current_epoch - best_epoch == 5:
                break
        """
        if auc > best_auc:
            best_auc = auc
            best_weights = model.get_weights()
            best_epoch = current_epoch
            best_pred = y_pred
        else:
            if current_epoch - best_epoch == 5:
                break



    model.set_weights(best_weights)
    return model, best_pred


def train_folds(X, y, fold_count, batch_size, get_model_func, aug = 1):
    size = len(X)//aug
    fold_size = size // fold_count
    models = []
    preds = []
    for fold_id in range(0, fold_count):
        print("Fold {}".format(fold_id))
        fold_start = fold_size * fold_id
        fold_end = fold_start + fold_size

        if fold_id == fold_count - 1:
            fold_end = size

        train_x = np.concatenate([X[:fold_start], X[fold_end:size]])
        train_y = np.concatenate([y[:fold_start], y[fold_end:size]])

        val_x = X[fold_start:fold_end]
        val_y = y[fold_start:fold_end]

        for aug_id in range(1, aug):

            train_x = np.concatenate([train_x, X[aug_id*size:aug_id*size + fold_start], X[aug_id*size + fold_end:(aug_id + 1)*size]])
            train_y = np.concatenate([train_y, y[aug_id*size:aug_id*size + fold_start], y[aug_id*size + fold_end:(aug_id + 1)*size]])

            # val_x = np.concatenate([val_x, X[aug_id*size + fold_start:aug_id*size + fold_end]])
            # val_y = np.concatenate([val_y, y[aug_id*size + fold_start:aug_id*size + fold_end]])

        model, pred = _train_model(get_model_func(), batch_size, train_x, train_y, val_x, val_y)
        models.append(model)
        preds.extend(pred)
    print("========================")
    print('Corssvalidation auc is {}'.format(roc_auc_score(y, preds)))
    return models, preds
