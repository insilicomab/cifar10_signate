# -*- coding: utf-8 -*-

# ライブラリの読み込み
import pandas as pd
import matplotlib.pyplot as plt

# ラベルマスターの読み込み
train_labels = pd.read_csv('./data/train_master.tsv', sep='\t')

# label_idの要素ごとの個数を確認
train_labels['label_id'].value_counts().plot(kind='bar')
plt.show()
