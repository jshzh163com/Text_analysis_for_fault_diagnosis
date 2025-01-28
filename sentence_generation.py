# -*- coding: utf-8 -*-


from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

from scipy import stats
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.model_selection import train_test_split
from CWRUdata import get_files

#   file path
data_dir = r'D:\Downloads\Mechanical-datasets-master\dataset'
list_data1 = get_files(data_dir, [0])

#   flatten the signals, shown in a row
signal_size = 1024
data = np.array(list_data1[0]).reshape(-1, signal_size)
label = np.array(list_data1[1])

#   data split
data_train, data_test, label_train, label_test = train_test_split(
    data, label, test_size=0.3, shuffle=True)

knn = KNN(n_neighbors=10, algorithm='auto',
          weights='distance', leaf_size=30,
          metric='minkowski', p=2,
          metric_params=None, n_jobs=1)

#   original data
train_knn = knn.fit(data_train, label_train)
train_knn.score(data_test, label_test)

'''
features extracted manually
'''
para_1 = np.mean(data, 1)
para_2 = np.std(data, 1)
para_3 = np.sqrt(np.mean(data**2, 1))
para_4 = np.array(list(map(max, abs(data))))
para_5 = stats.skew(data, 1)
para_6 = stats.kurtosis(data, 1)
para_7 = np.mean(abs(np.fft.fft(data)), 1)
para_8 = np.std(abs(np.fft.fft(data)), 1)
para_9 = np.sqrt(np.mean(abs(np.fft.fft(data))**2, 1))

para = np.vstack((para_1, para_2, para_3, para_4, para_5,
                 para_6, para_7, para_8, para_9)).T

# Define feature names
feature_names = ['Mean', 'Std', 'Sqrt', 'Maximum', 'Skewness',
                 'Kurtosis', 'Frequency mean', 'Frequency std', 'Frequency sqrt']

# Generate sentences for each sample
sentences = []
for row in para:
    sentence = ", ".join(
        [f"{feature} is {value:.2f}" for feature, value in zip(feature_names, row)])
    sentences.append(sentence)

# Combine sentences and labels
combined_data = [f"{sentence} Label: {label}" for sentence,
                 label in zip(sentences, label)]

# Save to a text file
output_file = "CWRU_sentences_with_labels_load_0.txt"
with open(output_file, 'w') as f:
    f.writelines("\n".join(combined_data))

# Print the first few lines to verify
for line in combined_data[:5]:
    print(line)
