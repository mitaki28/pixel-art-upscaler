＃ Pixel-Art Upscaler(仮)

このコードは[chaienr-pix2pix](https://github.com/pfnet-research/chainer-pix2pix)をベースにして作成されています。

![ドット絵の超解像]("https://raw.github.com/mitaki28/pixel-art-upscaler/image/example.png")

(変換元素材: [白螺子屋](http://hi79.web.fc2.com/)様, 学習データ: [First Seed Material](https://razor-edge.work/material/fsmchcv/) 様【閉鎖されてしまったので、代理配布先】）)

32x32〜16x16程度のキャラチップを前提としたドット絵の拡大ツールです。

(既存の手法)[https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms]よりもリアルな拡大が可能ですが、画像が歪んだり不自然な拡大がされることも結構あり、まだ実験段階です。

いわゆるディープラーニングと呼ばれる技術を用いて実装されており、[pix2pix](https://arxiv.org/abs/1611.07004) というネットワーク構造をベースにしています。実装は[chainer-pix2pix](https://github.com/pfnet-research/chainer-pix2pix)を改造して制作しました。

[こちら](https://razor-edge.work/material/fsmchcv/)で配布されている First Seed Material 様の素材（高解像度版）約7000枚を用いて学習しています。</p>
