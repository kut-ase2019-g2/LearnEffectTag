# LearnEffectTag



## Main

### LearnEffectTag.py

画像とSetTag.csv(エフェクトタグ)を関連付けてclarifaiにアップロードさせ学習を開始させるプログラム

### mymodel.py

学習させたモデルを実際に使用して、画像にエフェクトタグを対応付けたcsvを生成するプログラム



## Sub

### CheckData.py

EffectTagData(フォルダ)内の結果csvを結合するプログラム

### EffectDataMaker.py

CheckData.pyで生成したresult.csvファイルを元に実行

LearnEffectTag.pyで使用するcsvを生成するプログラム

### EffectCount.py

EffectDataMakerで生成したcsvファイルより、教師データの詳細を確認するプログラム

## Data

### EffectTagData(フォルダ)

5人による画像に対してエフェクトタグを選別したデータ

### result.csv

CheckData.pyで生成した5人の結果(EffectTagData)を統合したデータ

### SetTag.csv

result.csvを元に、EffectDataMaker.pyで生成したcsvファイル

2人以上選択されたタグを画像に関連づけるように設定されている



