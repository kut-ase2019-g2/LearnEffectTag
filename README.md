# LearnEffectTag



## Main

### LearnEffectTag.py

画像とcsv(エフェクトタグ)を関連付けてclarifaiにアップロードさせ学習を開始させるプログラム

### mymodel.py

学習させたモデルを実際に使用して、画像にエフェクトタグを対応付けたcsvを生成するプログラム



## Sub

### CheckData.py

複数人(今回は5人)の結果ファイルcsvを結合するプログラム

### EffectDataMaker.py

CheckData.pyで生成したcsvファイルを元に実行

LearnEffectTag.pyで使用するcsvを生成するプログラム

### EffectCount.py

EffectDataMakerで生成したcsvファイルより、教師データの詳細を確認するプログラム