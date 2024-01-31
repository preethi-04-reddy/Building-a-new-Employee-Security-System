#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import seaborn as sns
import matplotlib.pyplot as plt

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
data = pd.read_csv(r'C:\Users\91630\Downloads\archive (3)\train.csv', engine='python')
test_data = pd.read_csv(r'C:\Users\91630\Downloads\archive (3)\train.csv', engine='python')
print(data.head())


# In[38]:


data.Activity.unique()


# In[46]:


# Separate features and labels
X = data.drop('Activity', axis=1)
y = data['Activity']

# Perform t-SNE
tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300)
X_tsne = tsne.fit_transform(X)

# Create a DataFrame for the t-SNE results
tsne_df = pd.DataFrame({'X': X_tsne[:, 0], 'Y': X_tsne[:, 1], 'Activity': y})

# Plotting
plt.figure(figsize=(10, 6))
sns.scatterplot(x='X', y='Y', hue='Activity', data=tsne_df, palette='bright')
plt.title('t-SNE visualization of Human Activity Recognition Data')
plt.show()


# In[40]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler

X_train = data.drop(['Activity', 'subject'], axis=1)
y_train = data['Activity']

# Load test dataset
X_test = test_data.drop(['Activity', 'subject'], axis=1)
y_test = test_data['Activity']

# Preprocessing: Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[41]:


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.utils import to_categorical

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Encode the labels
encoder = LabelEncoder()
y_train_encoded = encoder.fit_transform(y_train)
y_test_encoded = encoder.transform(y_test)
y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Reshape input for LSTM [samples, time steps, features]
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))


# In[42]:


model = Sequential()
model.add(LSTM(100, input_shape=(X_train_reshaped.shape[1], X_train_reshaped.shape[2])))
model.add(Dropout(0.5))
model.add(Dense(y_train_categorical.shape[1], activation='softmax'))

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train_reshaped, y_train_categorical, epochs=20, batch_size=128, 
                    validation_data=(X_test_reshaped, y_test_categorical))

# Evaluate the model
loss, accuracy = model.evaluate(X_test_reshaped, y_test_categorical)
print("Accuracy: %.2f%%" % (accuracy * 100))


# In[43]:


plt.figure(figsize=(12, 6))

# Plot training & validation accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend()

# Plot training & validation loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()

plt.show()


# In[47]:


# Make predictions
y_pred = model.predict(X_test_reshaped)

# Convert one-hot encoded predictions back to class labels
y_pred_classes = np.argmax(y_pred, axis=1)

# Convert one-hot encoded test labels back to class labels
y_test_classes = np.argmax(y_test_categorical, axis=1)
from sklearn.metrics import confusion_matrix
# Compute confusion matrix
cm = confusion_matrix(y_test_classes, y_pred_classes)

# Fetch labels for plotting
class_labels = encoder.classes_

# Plot the confusion matrix using Seaborn
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_labels, yticklabels=class_labels)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix for LSTM Model')
plt.show()


# In[ ]:




