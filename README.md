## Pixcaler(Pixel-art scaler)

* Demo： http://mitaki28.info/pixcaler/

<img src="https://github.com/mitaki28/pixel-art-upscaler/blob/master/image/example.gif?raw=true">

(変換元素材: [白螺子屋](http://hi79.web.fc2.com/)様, 学習データ: [カミソリエッジ](https://razor-edge.work/material/fsmchcv/) 様【オリジナルの素材を配布していたのは First Seed Material 様（サイト閉鎖）】）

ドット絵に特化した拡大ツールです。

[既存の手法](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms)と比較してより自然な拡大が可能です。

いわゆるディープラーニングと呼ばれる技術を用いて実装されており、[pix2pix](https://arxiv.org/abs/1611.07004) というネットワーク構造をベースにしています。実装は[chainer-pix2pix](https://github.com/pfnet-research/chainer-pix2pix)を改造して制作しました。

* 以下の素材を機械的に重ね合わせて合成したデータを用いて学習しています
    * [カミソリエッジ](https://razor-edge.work/material/fsmchcv/)様が配布されている First Seed Material 素材（高解像度版）のカラーバリエーション約7000枚
    * [M+フォント](https://mplus-fonts.osdn.jp/)全種から、light, thin を除いたもの
    * [コミュ将](https://comshou.wixsite.com/com-sho/about)様の配布されている[タイルセット（RTP不使用版）](https://comshou.wixsite.com/com-sho/single-post/2017/04/19/RTP%E4%B8%8D%E4%BD%BF%E7%94%A8%E7%B4%A0%E6%9D%90%E3%81%BE%E3%81%A8%E3%82%81)

### 環境構築
* Python 3.5 が必要です

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

* GPU を利用する場合は手動で cupy を追加インストールしてください
```
pip install cupy
```

### データセット
* データセットを用意するディレクトリを `image/dataset` とします。
    * `image/dataset/chartip` に学習に使いたいキャラチップ素材を **背景を透過して** png 形式で保存します
        * 背景を透過するためのユーティリティを用意しています。
            ```
            python -m pixcaler.tool.transparent-chartip image/dataset/chartip /path/to/chartip1.png /path/to/chatip2.png
            ```
    * `image/dataset/tile` に学習に使いたいマップチップ素材のうちタイル用の素材を png 形式で保存します
    * `image/dataset/obj` に学習に使いたいマップチップ素材のうち前景に配置するオブジェクト素材を png 形式で保存します
    * `image/dataset/font` に学習に使いたいフォント素材を ttf 形式で保存します
    * `image/dataset/test` に学習結果のプレビュー用に拡大する画像を png 形式で保存します
        

### 学習
1. 以下のコマンドを実行します(GPU利用推奨)
    * GPUを利用しない場合、数週間〜数ヶ月学習にかかる可能性があります
```
python -m pixcaler.train -g [GPU ID or -1(CPUを使う場合)]--composite -i (データセットのディレクトリ) -b 4
```


### 学習したモデルを使った画像変換
1. 上記の手順に従った場合、通常、 `result/gen_iter_{iteration}.npz` に世代ごとのモデルが出力されます。iteration には学習のイテレーション回数（数値列）が入ります。
1. 以下のコマンドを実行します。
```
python -m pixcaler.run --generator=result/gen_iter_{iteration}.npz --mode up /path/to/image1.png /path/to/image2.png
```


### 既存の実装からの変更点
#### 学習データ
* もとの画像を以下の手順で処理して学習しています
    * キャラチップに関しては点(0, 0) の色を透明色とみなし、該当する色を(0, 0, 0, 0)に変更します
    * キャラチップから64x64の範囲をランダムに切り取ります
    * キャラチップ、フォントを背景のタイルセットからランダムにまたは固定色1色の背景と合成します
    * 画像を平行移動（64x64のcrop）と鏡像反転して、データを水増しします
        * 平行移動はおそらく重要です
            * nearest neighbor 縮小は性質上、1枚の画像に対して、4種類の結果が存在します（縮小時に4x4格子のどの点を取るかの自由度があるため）
                * <img src="https://github.com/mitaki28/pixel-art-upscaler/blob/master/nn-scales.png?raw=true">
            * 画像をランダムに平行移動することによって、この4種類の結果をすべて学習データセットに加えることができ、実質的に画像を4倍に水増しできます
            * さらに、目などの細かいピクセルの模様も nearest neighbor 縮小のいずれかのパターンではきれいに残っていること多いので、学習を繰り返すことで、適切な復元方法にたどり着ける可能性が上がります
    * nearest neighbor 法(PIL.Image.NEAREST_NEIGHBOR によるリサイズ)で縮小し、再度64x64に nearest_neighbor 法で拡大したものを変換元、もとの画像を変換先として学習します

#### その他
* Generater/Discriminater の loss 関数を [LSGAN](https://arxiv.org/abs/1611.04076) に変更(効果があるかは微妙)
    * 同時に lam1 倍率を100→10に変更してます（経験上、lsgan に換装すると loss は10分の1ぐらいにスケールされる）
    * [CycleGAN](https://github.com/junyanz/CycleGAN) でも採用されているより安定性の高い loss 関数
* adversarial loss 倍率を1/16に変更
    * ドット絵の場合、l1-lossが通常の写真などよりもより小さい値に収束するため、adversarial loss をかなり小さく取らないと学習が不安定になります
    * なお、この倍率だと、loss の値上は、adversarial loss がほとんど無視されているような挙動になりますが、完全に adversarial loss をなくしてしまうと、出力にノイズが乗るようになり、学習結果が不安定になります
* pix2pix ネットワークの encoder, decoder の最上段を kernel size 5x5, stride 1, padding 2 の Convolution2D に換装（効果あるのか微妙）
    * もとのネットワークでは画像サイズが128x128以上ないと、画像幅が足りずエラーになります
    * そこで、最上段を5x5のConvolution2D(縮小なし)に換装しました
    * 3x3 ではなく5x5 なのは [既存の手法](https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms) が5x5のconvolutionをベースとしていたことや、より広い範囲を見たほうが、そのドットのコンテキストを推論しやすいだろうという予想のもとです


