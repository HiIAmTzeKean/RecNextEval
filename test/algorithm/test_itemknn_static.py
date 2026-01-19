from recnexteval.algorithms import ItemKNNStatic


def test_ItemKNNStatic(setting):
    algo = ItemKNNStatic(K=10)
    setting.training_data.mask_shape()
    algo.fit(setting.training_data)
    unlabeled_data = setting.unlabeled_data[0]
    unlabeled_data.mask_shape(setting.training_data.shape, True, True)
    X_pred = algo.predict(unlabeled_data)
