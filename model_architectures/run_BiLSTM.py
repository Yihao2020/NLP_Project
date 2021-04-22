from  model_architectures import BiLSTM

model = BiLSTM.BiLSTM1(epochs = 10)
model.train("embedding.pkl")
model.evaluate()
model.plot_metrics()
model = BiLSTM.BiLSTM2(epochs = 10)
model.train("embedding.pkl")
model.evaluate()
model.plot_metrics()