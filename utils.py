import pandas as pd
import numpy as np

def load_dataset(dataset="https://raw.githubusercontent.com/OptativoPUCV/Fashion-DataSet/master/fashion-1.csv"):
    data = pd.read_csv(dataset)
    label = data.label.to_numpy()
    data.drop("label",inplace=True, axis=1)

    data = data.to_numpy()
    data = data.astype(float) / 255. ## Todos los pixeles se normalizan

    per = int(data.shape[0] * 0.7) # se ocupa el 70% para entrenar el resto para test
    data_split = np.split(data, (per,))
    label_split = np.split(label, (per,))

    data, X_val = data_split[0], data_split[1]
    label, y_val = label_split[0], label_split[1]

    return data, label, X_val, y_val