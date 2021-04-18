# -*- coding: utf-8 -*-

'''
データの読み込みと確認
'''

# ライブラリのインポート
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from skimage import io
import os
import zipfile
import io

from sklearn.model_selection import KFold

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint

# ランダムシードの設定
import random
np.random.seed(1234)
random.seed(1234)
tf.random.set_seed(1234)

# 入力と出力を指定
photo_size = 32
im_rows = 32 # 画像の縦ピクセルサイズ
im_cols = 32 # 画像の横ピクセルサイズ
im_color = 3 # 画像の色空間/グレイスケール
in_shape = (im_rows, im_cols, im_color)
num_classes = 10 # クラス数の定義:4クラス

# ラベルデータの読み込み
train_labels = pd.read_csv('./data/train_master.tsv', sep = '\t')
print(train_labels.head())

# zipファイルのパス
zip_path = './data/train_images.zip'

# 画像データの格納リスト
X_train = [] # 画像のピクセル値とラベルを格納するリストを生成（説明変数）

# zipの読み込み
with zipfile.ZipFile(zip_path, 'r') as zip_file:
    for i in train_labels['file_name']:
        with zip_file.open('train_images/'+i) as img_file:
            # 画像のバイナリデータを読み込む
            img_bin = io.BytesIO(img_file.read())
            # バイナリデータをpillowで開く
            img = Image.open(img_bin)
            # サイズ変更
            img = img.resize((photo_size, photo_size))
            # 画像データを配列化
            img_array = np.array(img)
            # 格納リストに追加
            X_train.append(img_array)

# 画像の確認
for i in range(10):
    plt.imshow(X_train[i])
    plt.show()

# np.arrayに変換
X_train = np.array(X_train)

'''
特徴量エンジニアリング
'''

# 読み込んだデータを三次元配列に変換後、正規化
X_train = X_train.reshape(-1, im_rows, im_cols, im_color)
X_train = X_train.astype('float32') / 255

# ラベルデータのOne-hot encoding
Y_train = np.array(train_labels['label_id'])
Y_train = keras.utils.to_categorical(Y_train, num_classes)

'''
モデリングと評価
'''

# K分割する
folds = 5
kf = KFold(n_splits=folds)
index = 0

for train_index, val_index in kf.split(X_train):
    x_train = X_train[train_index]
    x_valid = X_train[val_index]
    y_train = Y_train[train_index]
    y_valid = Y_train[val_index]
    
    # モデルを保存するファイルパス
    filepath = './model/cifar10_CNN_model[%d].h5' % index
    index += 1
    
    # CNNモデルを定義
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same',
                     input_shape=in_shape))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes))
    model.add(Activation('softmax'))
    
    # モデルをコンパイル
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])
    
    # 過学習の抑制
    early_stopping = EarlyStopping(monitor='val_loss', patience=10 , verbose=1)
    
    # 評価に用いるモデル重みデータの保存
    checkpointer = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
    
    # 学習を実行
    hist = model.fit(x_train, y_train,
                     batch_size=128, epochs=30,
                     verbose=1,
                     validation_data=(x_valid, y_valid),
                     callbacks=[early_stopping, checkpointer])  # CallBacksに設定
    
    # モデルを評価
    score = model.evaluate(x_valid, y_valid, verbose=1)
    print('正解率=', score[1], 'loss=', score[0])
    
    '''
    学習過程のグラフ化
    '''
    
    # 正解率の推移をプロット
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.title('Accuracy')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

    # ロスの推移をプロット
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.title('Loss')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

# 評価に用いるモデル構造の保存
def save_model(model):
    json_string = model.to_json()
    if not os.path.isdir("cache"):
        os.mkdir("cache")
    json_name = "architecture.json"
    open(os.path.join("cache", json_name),"w").write(json_string)

save_model(model)