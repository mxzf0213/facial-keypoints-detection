from keras.models import load_model
import os
from DataProcess import load2d
import csv

if os.path.exists('CnnModel.h5'):
    model = load_model('CnnModel.h5')
else:
    print("Error! There is no trained model!")
    exit(0)

x_test, _ = load2d(test=True, cols=False)
result = model.predict(x_test)

with open('result.csv', 'w', encoding='utf-8', newline='') as f:
    csv_writer = csv.writer(f)
    csv_writer.writerows(result)