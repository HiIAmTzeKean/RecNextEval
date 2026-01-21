from recnexteval.algorithms import ItemKNNIncremental

def test_ItemKNNIncremental(setting):
    algo = ItemKNNIncremental(K=10)
    setting.training_data.mask_shape()
    algo.fit(setting.training_data)
    unlabeled_data = setting.unlabeled_data[0]
    unlabeled_data.mask_shape(setting.training_data.shape, True, True)
    X_pred = algo.predict(unlabeled_data)