# TensorFlowとAndroidによる手書きのハングル文字の認識

*Read this in other languages: [English](README.md),[韓国語](README-ko.md).*

アルファベットに相当する韓国語のハングル文字には、19 個の子音と 21 個の母音があります。これらの子音と母音を組み合わせて構成できるハングルの音節と文字は合計で 11,172 個にも上ります。ただし、通常使われているのは、そのほんの一部です。

このパターンでは、独自の韓国語トレーニング・データを生成するプロセスを説明した後、TensorFlow モデルをトレーニングして、手書きの、一般的なハングル文字を分類できるようにします。次に、ユーザーがモバイル・デバイス上で書いた韓国語の文字を、トレーニング済みモデルを使って認識する Android アプリケーションを作成し、実行します。このアプリケーション内で韓国語の単語または文を書くと、[Watson Language Translator](https://www.ibm.com/watson/jp-ja/developercloud/language-translator.html) サービスによってそれを翻訳することができます。

![Demo App](doc/source/images/hangul_tensordroid_demo1.gif "Android application")

今回のパターンは以下を含みます:
1. ハングルをサポートしている無料フォントをオンラインで探し、学習用の画像データを生成します。
2. 画像を TFRecords 形式に変換し、モデルの入力と訓練に使用します。
3. モデルをトレーニングし保存します。
4. シンプルな Android アプリケーションで保存されたモデルを使用します。
5. Watson Language Translator サービスを接続して文字を翻訳します。

![architecture](doc/source/images/architecture.png)

## Flow

1. データを生成するために使用する韓国語のフォントをいくつかダウンロードします。
2. フォントから生成された画像が、トレーニング対象の TensorFlow モデルに取り込まれます。
3. ユーザーが Android デバイス上で韓国語の文字を書きます。
4. トレーニング済み TensorFlow モデルと Android TensorFlow Inference インターフェースを使用して、手書きの文字が識別されます。
5. 分類された韓国語の文字列が Watson Language Translator サービスに送信されて、英語に翻訳されます。

## 含まれるコンポーネント

* [Watson Language Translator](https://www.ibm.com/watson/jp-ja/developercloud/language-translator.html): コンテンツのテキストを、ある言語から別の言語にリアルタイムで翻訳する、ドメインに最適化された IBM Cloud 上のサービスです。
* [TensorFlow](https://www.tensorflow.org/): 機械学習用のオープンソースソフトウェアライブラリ。
* [Android](https://developer.android.com/develop/index.html): Linuxカーネルに基づくオープンソースのモバイルオペレーティングシステム。

## 利用した技術

* [Artificial Intelligence](https://developer.ibm.com/jp/technologies/artificial-intelligence/):
人間のような理解、理性、学び、そして相互作用ができる認知技術。
* [Mobile](https://developer.ibm.com/jp/technologies/mobile/): モバイルユーザー向けに特別に設計されたアプリを開発し、問題の解決を可能にする環境。

# ビデオを観る

[![](https://img.youtube.com/vi/iefYaCOz00s/0.jpg)](https://www.youtube.com/watch?v=iefYaCOz00s)

# 手順

## ローカルで実行する

このコードパターンを設定して実行するには、次の手順に従います。それぞれの手順については、後で詳しく説明します。

1. [リポジトリをクローンする](#1-clone-the-repo)
2. [必要なソフトウェアのインストール](#2-install-prerequisites)
3. [画像データの生成](#3-generate-image-data)
4. [画像を TFRecords に変換する](#4-convert-images-to-tfrecords)
5. [モデルをトレーニングする](#5-train-the-model)
6. [モデルを試す](#6-try-out-the-model)
7. [Android アプリの作成](#7-create-the-android-application)

<a name="1-clone-the-repo"></a>
### 1. リポジトリをクローンする

`tensorflow-hangul-recognition` をローカルにクローンします。ターミナルで、次のコマンドを実行します。

```
git clone https://github.com/IBM/tensorflow-hangul-recognition
```

<a name="2-install-prerequisites"></a>
### 2. 必要なソフトウェアのインストール

クローンしたリポジトリのフォルダーに移動します。
```
cd tensorflow-hangul-recognition
```

オプションで、ランタイム環境を分離するには、[こちら](https://packaging.python.org/guides/installing-using-pip-and-virtualenv/) を参照して仮想環境を使用します。仮想環境の作成:
```
python -m pip install --user virtualenv
python -m virtualenv .venv
source .venv/bin/activate
```

このコードパターンのために必要な Python 用のライブラリを一括インストールします：
```
pip install -r requirements.txt
```

コードパターンを完成したら、`deactivate` コマンドを使用して仮想環境を終了します。

**Note:** Windows環境の方は、_scipy_ パッケージを **pip** 経由でインストールできません。_scipy_ を使用するために推奨する方法は、
[特定の Python ディストリビューション](https://www.scipy.org/install.html#scientific-python-distributions) をインストールすることです。有名なものとしては [Anaconda](https://www.anaconda.com/download/) があります。もしくは [こちら](http://www.lfd.uci.edu/%7Egohlke/pythonlibs/#scipy) で紹介されている Windows 用のインストーラで手動インストールも可能です。

<a name="3-generate-image-data"></a>
### 3. 画像データの生成

優れたモデルを訓練するためには、膨大な量のデータが必要です。しかし、実際の手書き韓国文字のデータセットを十分に取得することは、見つけるのが難しく、また作成が面倒です。

このデータの問題に対処する1つの方法は、オンラインで見つかった豊富な韓国語フォントファイルを利用して、プログラムで画像データを自動生成することです。これが、まさに私たちが今回やることです。

このリポジトリの tools ディレクトリにあるのが [hangul-image-generator.py](./tools/hangul-image-generator.py) です。
このスクリプトは、フォントディレクトリにあるフォントを使用して、指定されたラベルファイルに含まれる各文字のイメージを作成します。

デフォルトのラベルファイルは [2350-common-hangul.txt](./labels/2350-common-hangul.txt) で、
[KS X 1001 encoding](https://en.wikipedia.org/wiki/KS_X_1001) から得られた2350の頻繁に使用される文字を含んでいます
他のラベルファイルは [256-common-hangul.txt](./labels/256-common-hangul.txt) と [512-common-hangul.txt](./labels/512-common-hangul.txt) です。
これらは、韓国語の国立研究所のサイトに掲載されている [上位6000個の韓国語](https://www.topikguide.com/download/6000_korean_words.htm) から作成されました。
もし訓練するマシンのパワーが十分でない場合、小さなラベルセットを使用することで後でモデルトレーニングの時間を減らすことができます。

[fonts](./fonts) フォルダは現在空です。したがって、ハングルデータセットを生成するには、まずフォントディレクトリにある [README](./fonts/README.md) の説明に従っていくつかのフォントファイルをダウンロードする必要があります。
私のデータセットでは、約40種類のフォントファイルを使用しましたが、より多くのフォントファイルを使用してデータセットを改善することができます。ユニークなスタイルのものが特にお勧めです。
フォントディレクトリが準備できたら、[hangul-image-generator.py](./tools/hangul-image-generator.py) を使って画像生成を実施できます。

実行時のオプションは次のとおりです:

* `--label-file` 別の(たぶんより小さな)ラベルファイルを指定します。初期値は _./labels/2350-common-hangul.txt_ です。
* `--font-dir` 別のフォントディレクトリを指定します。初期値は _./fonts_ です。
* `--output-dir` 別の生成された画像の保存先を指定します。初期値は _./image-data_ です。

ラベルファイルを指定して実行してみましょう:
```
python ./tools/hangul-image-generator.py --label-file <your label file path>
```

ラベルとフォントの数に応じて、このスクリプトは完了するまでに時間がかかることがあります。
データセットを強化するために、生成された各文字画像に対して3つのランダムな弾性ひずみも実行されます。
以下がその例で、最初に表示されているのが元の文字で、それ以降に弾性ひずみが付与されたものが続きます。

![Normal Image](doc/source/images/hangul_normal.jpeg "Normal font character image")
![Distorted Image 1](doc/source/images/hangul_distorted1.jpeg "Distorted font character image")
![Distorted Image 2](doc/source/images/hangul_distorted2.jpeg "Distorted font character image")
![Distorted Image 3](doc/source/images/hangul_distorted3.jpeg "Distorted font character image")

スクリプトが完了すると、出力ディレクトリには、すべての64x64 JPEG画像を保持する _hangul-images_ フォルダが作成されます。
出力ディレクトリには、すべての画像パスを対応するラベルにマップする _labels-map.csv_ ファイルも作成されます。

<a name="4-convert-images-to-tfrecords"></a>
### 4. 画像を TFRecords に変換する

TensorFlow の標準入力フォーマットは [TFRecords](https://www.tensorflow.org/api_guides/python/python_io#tfrecords_format_details) で、これは生のイメージデータとそのラベルを一緒に保存するバイナリ形式です。
TensorFlow モデルへのデータフィードをより良くするために、まず画像からいくつかの TFRecords ファイルを作成しましょう。
これを行うための [スクリプト](./tools/convert-to-tfrecords.py) が用意されています。

このスクリプトは、さきほど生成した _labels-map.csv_ ファイルに基づいて、すべてのイメージとラベルデータを読み込みます。
次に、トレーニングセットとテストセット (テスト15%、トレーニング85%) を持つようにデータを分割します。
デフォルトでは、トレーニングセットは複数のファイル/シャード(3つ)に保存され、1つの巨大なファイルにならないようにしますが、これはCLIの引数 _--num-shards-train_ でデータセットサイズに応じて設定できます。

スクリプト実行時のオプションは次のとおりです:

* `--image-label-csv` 画像パスをラベルにマップするCSVファイルを指定する。初期値は _./image-data/labels-map.csv_ です。
* `--label-file` トレーニングセットに対応するラベルを指定する。スクリプトによってクラスの数を決定するために使用されます。初期値は _./labels/2350-common-hangul.txt_ です。
* `--output-dir` 生成された TFRecords ファイルの出力ディレクトリを指定する。初期値は _./tfrecords-output_ です。
* `--num-shards-train` トレーニングセット TFRecords を分割する分割数を指定する。初期値は _3_ です。
* `--num-shards-test` テストセット TFRecords を分割する分割数を指定する。初期値は _1_ です。

スクリプトを以下のように実行します。
```
python ./tools/convert-to-tfrecords.py --label-file <your label file path>
```

このスクリプトが完了したら、出力ディレクトリ _./tfrecords-output_ に TFRecords ファイルが生成されていることを確認してください。

```
$ ls ./tfrecords-output
test1.tfrecords    train1.tfrecords    train2.tfrecords    train3.tfrecords
```

<a name="5-train-the-model"></a>
### 5. モデルをトレーニングする

多くのデータを準備できたので、実際にそれを使用しましょう。
プロジェクトのルートに [hangul_model.py](./hangul_model.py) があります。
このスクリプトは、TFRecords ファイルを読み込み、画像とラベルのランダムな束 (バッチ) を提供するための入力パイプラインを作成します。
次に、畳み込みニューラルネットワーク (CNN: Convolutional Neural Network) が定義され、トレーニングが実行されます。
トレーニングプロセスは、CNNに画像とラベルの束を連続的に送り、各キャラクタを正しく分類するための最適な重みと偏りを見つけます。
トレーニング後、モデルは Android アプリケーションで使用できるようにエクスポートされます。

ここで使用しているモデルは、[TensorFlow](https://www.tensorflow.org/get_started/mnist/pros) に記載されているMNISTモデルに似ています。
第3の畳み込みレイヤーが追加され、さらに多くのクラスを分類するのに役立つ多くの機能が追加されています。

スクリプト実行時のオプションは次のとおりです:

* `--label-file` トレーニングセットに対応するラベルを指定する。スクリプトによってクラスの数を決定するために使用されます。初期値は _./labels/2350-common-hangul.txt_ です。
* `--tfrecords-dir` TFRecords シャードを出力する格納するディレクトリを指定します。初期値は _./tfrecords-output_ です。
* `--output-dir` 生成されたモデルのチェックポイント、グラフ、プロトコルバッファーファイルの出力ディレクトリを指定する。初期値は _./saved-model_  です。
* `--num-train-steps` 実行するトレーニングステップの数を指定する。
  これは、より多くのデータで増加させる必要があります(またはその逆)。
  ステップ数は、すべてのトレーニングデータ(エポック)にわたり数回の反復をカバーする必要があります。
  例えば、トレーニングセットに 320,000 の画像がある場合、_100_ がデフォルトのバッチサイズならば、_320000 / 100 = 3200_ ステップが1エポックになります。
  だから、もし30エポック訓練したいのであれば、単に _3200 * 30 = 96000_ のトレーニングステップを指定します。
  このパラメータを自分で調整して、少なくとも15エポックを試してみてください。
  初期値は _30000_ ステップです。

トレーニングを実行するには、プロジェクトのルートから次の操作を行います:
```
python ./hangul_model.py --label-file <your label file path> --num-train-steps <num>
```

対象の画像の数によっては、特にラップトップでのトレーニングの場合、トレーニングには時間がかかります (数時間、1日を超えることも)。
もし GPU へのアクセス権があるならば、GPU をサポートする TensorFlow をインストールすることで、これらの作業が高速化されます。
([Ubuntu](https://www.tensorflow.org/install/install_linux) と [Windows](https://www.tensorflow.org/install/install_windows) のみサポート)

Nvidia GTX 1080グラフィックカードを搭載した私の Windows デスクトップコンピュータでは、スクリプトのデフォルト設定で約 320,000 の画像をトレーニングするのに2時間ちょっとしかかかりませんでした。
私の MacBook Pro でのトレーニングは、おそらく20倍以上かかるでしょう。

1つの選択肢として、縮小されたラベルセット (例えば 2350 ではなく 256 ハングル文字) を使用することが考えられ、計算の複雑さを相当に低減することができます。

スクリプトが実行されるにつれて、表示される訓練の精度が 1.0 に向かうことが期待できます。
また、訓練の後のテストにも、適切な精度が必要です。
スクリプトが完了すると、使用のためエクスポートされたモデルは `./saved-model/optimized_hangul_tensorflow.pb` として保存されます。
これは学習したすべての重みと偏りを持つモデルをシリアライズ化して保存した [プロトコルバッファー](https://ja.wikipedia.org/wiki/Protocol_Buffers) ファイルです。
このファイルは、推論のみの使用のために最適化されています。

<a name="6-try-out-the-model"></a>
### 6. モデルを試す

新しく保存したモデルで Android アプリケーションを作成する前に、まずそれを試してみましょう。
あなたのモデルをロードし、与えられた画像の推論を試してみる [スクリプト](./tools/classify-hangul.py) が提供されています。
自分の画像で試してみるか、下のサンプル画像をダウンロードしてください。
各画像が黒色の背景と白の文字色で、サイズが 64x64 ピクセルであることを確認してください。

スクリプト実行時のオプションは次のとおりです:

* `--label-file` ラベルを指定する。これは、One-hotラベル表現のインデックスを実際の文字にマッピングするために使用されます。初期値は _./labels/2350-common-hangul.txt_ です。
* `--graph-file` 保存されたモデルファイルを指定します。初期値は _./saved-model/optimized_hangul_tensorflow.pb_ です。

以下のように実行します:

```
python ./tools/classify-hangul.py <Image Path> --label-file <your label file path>
```

***サンプル画像:***

![Sample Image 1](doc/source/images/hangul_sample1.jpeg "Sample image")
![Sample Image 2](doc/source/images/hangul_sample2.jpeg "Sample image")
![Sample Image 3](doc/source/images/hangul_sample3.jpeg "Sample image")
![Sample Image 4](doc/source/images/hangul_sample4.jpeg "Sample image")
![Sample Image 5](doc/source/images/hangul_sample5.jpeg "Sample image")

スクリプトを実行すると、上位5つの予測とそれに対応するスコアが表示されます。
うまくいけば、一番上の予測はあなたのキャラクターの実際のものとマッチします。

**ノート**: このスクリプトを Windows で実行している場合、韓国語の文字をコンソールに表示するには、まずアクティブなコードページを UTF-8 をサポートするように変更する必要があります。以下を実行します：

```
chcp 65001
```

次に、コンソールのフォント設定を Batang、Dotum、Gulim などの韓国語のテキストをサポートするように変更する必要があります。

<a name="7-create-the-android-application"></a>
### 7. Android アプリの作成

保存されたモデルを使用すると、ユーザーが作成した手書きのハングルを分類できるシンプルな Android アプリケーションを作成できます。
完成したアプリケーションは [./hangul-tensordroid](./hangul-tensordroid) にあります。

#### プロジェクトの準備

アプリ作成を自分で試してみる最も簡単な方法は、[Android Studio](https://developer.android.com/studio/index.html) を使用することです。
この統合開発環境(IDE)は、自動で多くの Android 依存関係を処理してくれます。

Android Studio をダウンロードしてインストールしたら、次の手順を実行します:

1. Android Studio を起動する
2. **Welcome to Android Studio** ウインドウが表示されるので、**Open an existing Android Studio project** をクリック。もしこのウインドウが表示されなければ、トップメニューから **File > Open...** を選択します。
3. ファイル選択画面でこのプロジェクトの _./hangul-tensordroid_ ディレクトリに移動し、 **OK** ボタンを押す。

ビルドと初期化の後、プロジェクトは Android スタジオから利用できるようになります。
Gradle が初めてプロジェクトをビルドするとき、いくつかの依存関係の問題があるかもしれません。
これらは Android Studio では、表示されたエラープロンプトの「依存関係をインストール」リンクをクリックすることで、簡単に解決できます。

Android Studio では、サイドメニューからプロジェクトの構造を簡単に確認できます。

![Android Project Structure](doc/source/images/android-project-structure.png "Project Structure")

java フォルダには、アプリケーションのすべての Java ソースコードが含まれています。
これを展開すると、Javaファイルが4つしかないことがわかります:

1. **[MainActivity.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/MainActivity.java)**
   はアプリケーションの主な起動ポイントであり、セットアップとボタン押下ロジックを処理します。
2. **[PaintView.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/PaintView.java)**
   は、ユーザーが画面上のビットマップに韓国語の文字を描くことを可能にするクラスです。
3. **[HangulClassifier.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/HangulClassifier.java)**
   は事前に訓練されたモデルをロードし、それを TensorFlow 推論インタフェースに接続して、分類のために画像を渡すことができます。
4. **[HangulTranslator.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/HangulTranslator.java)**
   は、Watson Language Translator APIとのインターフェイスを使用して、テキストを英語に翻訳します。

現在の状態では、提供されている Android アプリケーションは _2350-common-hangul.txt_ ラベルファイルを使用しており、既に 40 のフォントから約 320,000 の画像で訓練された訓練済みのモデルを持っています。
これらはプロジェクトの _assets_ フォルダ、_./hangul-tensordroid/app/src/main/assets/_ にあります。

モデルファイルまたはラベルファイルを切り替えるには、単にこのディレクトリに配置します。
_./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/_ ディレクトリにある [MainActivity.java](./hangul-tensordroid/app/src/main/java/ibm/tf/hangul/MainActivity.java) ファイルに配置したファイルの名前を指定する必要があります。
クラスの先頭にある定数 `LABEL_FILE` と `MODEL_FILE` の値を変更するだけです。

翻訳サポートを有効にするには、次の操作を行う必要があります:

1. まだ所有していない場合、[こちら](https://cloud.ibm.com/registration/) で IBM Cloud アカウントを作成する。
2. [Watson Language Translator](https://cloud.ibm.com/catalog/services/language-translator)
   サービスを作成する。
3. Translator サービスの資格情報を入手する。
資格情報は自動的に作成されています。
自身の IBM Cloud ダッシュボードの **Services** セクションの下にある **Language Translator** サービスをクリックすることで資格情報を確認できます。
4. ステップ3で入手した **username** と **password** を [./hangul-tensordroid/app/src/main/res/values/translate_api.xml](./hangul-tensordroid/app/src/main/res/values/translate_api.xml) ファイルに反映する。

#### アプリの実行

アプリケーションをビルドして実行する準備ができたら、Android スタジオの上部にある緑色の矢印ボタンをクリックします。

![Android Studio Run Button](doc/source/images/android-studio-play-button.png "Run Button")

**Select Deployment Target** ウインドウが表示されるはずです。

実際の Android 搭載端末をお持ちの場合は、USB を使用してパソコンに接続してください。
詳細は [こちら](https://developer.android.com/studio/run/device.html) をご参照ください。

Android デバイスをお持ちでない場合は、代わりにエミュレータを使用することもできます。
**Select Deployment Target** ウィンドウで、**Create New Virtual Device** をクリックします。
次に、ウィザードに従い、デバイス定義とイメージ (APIレベル21以上のイメージが望ましい) を選択します。
仮想デバイスが作成されたら、アプリケーションを実行するときに仮想デバイスを選択できるようになります。

デバイスを選択すると、アプリケーションは自動的にビルド、インストールされ、その対象デバイスで起動します。

アプリケーション上でドローイングして、モデルがハングルの文字をどれだけうまく認識しているかを確認してください。

# リンク

* [Deep MNIST for Experts](https://www.tensorflow.org/get_started/mnist/pros): 手書きの数字を認識するための畳み込みニューラルネットワークの作成とトレーニングのためのチュートリアル。
* [TensorFlow Mobile](https://www.tensorflow.org/mobile/): 異なるプラットフォーム上の TensorFlow モバイルサポートに関する情報。
* [Hangul Syllables](https://en.wikipedia.org/wiki/Hangul_Syllables): すべてのハングルの音節のリスト。

# もっと学ぶ

* **Artificial Intelligence コードパターン**: このコードパターンを気に入りましたか？ [AI コードパターン](https://developer.ibm.com/jp/technologies/artificial-intelligence/) から関連パターンを参照してください。
* **AI and Data コードパターン・プレイリスト**: コードパターンに関係するビデオ全ての [プレイリスト](https://www.youtube.com/playlist?list=PLzUbsvIyrNfknNewObx5N7uGZ5FKH0Fde) です。
* **With Watson**: [With Watson](https://www.ibm.com/watson/jp-ja/with-watson/) プログラム は、自社のアプリケーションに Watson テクノロジーを有効的に組み込んでいる開発者や企業に、ブランディング、マーケティング、テクニカルに関するリソースを提供するプログラムです。


# ライセンス

[Apache 2.0](LICENSE)
