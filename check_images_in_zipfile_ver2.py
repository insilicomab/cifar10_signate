# -*- coding: utf-8 -*-
'''
zipファイル内の画像ファイルを確認
'''

# ライブラリのインポート
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import zipfile
import io

# zipファイルのパス
zip_path = './data/train_images.zip'

# zipの中身を確認
with zipfile.ZipFile(zip_path, 'r') as zip_file:
    for info in zip_file.infolist():
        print(info)

'''
zipファイル内の画像ファイルの読み込み
'''

# ラベルデータの読み込み
train_labels = pd.read_csv('./data/train_master.tsv', sep='\t')
print(train_labels.head())

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
            # 画像データを配列化
            img_array = np.array(img)
            # 格納リストに追加
            X_train.append(img_array)

# ラベルマスタの読み込み
label_master = pd.read_csv('./data/label_master.tsv', sep='\t')
print(label_master.head())

# ラベル名を抽出
labels = label_master['label_name']

# 画像の確認
plt.figure(figsize=(10,10))
for i in range(0,40):
    plt.subplot(5,8,i+1)
    plt.title(labels[train_labels.iloc[i, 1]]) # train_labelsのラベルID抽出後、IDに対応するラベル名を抽出
    plt.axis('off') # 軸と目盛りをオフ
    plt.imshow(X_train[i])