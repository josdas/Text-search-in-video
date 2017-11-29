class NNet(dict):
    def __init__(self,
                 net,
                 train_fun,
                 loss_fun,
                 loss_fun_det,
                 predict_fun_det,
                 learning_rate):
        super().__init__(net)
        self.learning_rate = learning_rate
        self.predict_fun_det = predict_fun_det
        self.loss_fun_det = loss_fun_det
        self.loss_fun = loss_fun
        self.train_fun = train_fun
