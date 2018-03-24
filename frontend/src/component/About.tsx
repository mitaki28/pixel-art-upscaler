import * as React from "react";

export const About = (props: {}) => (
    <div>
        <p>32x32〜16x16程度のキャラチップを前提としたドット絵の拡大ツールです。</p>
        <p><a href="https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms">既存の手法</a>よりもリアルな拡大が可能ですが、画像が歪んだり不自然な拡大がされることも結構あり、まだ実験段階です。</p>
        <p>いわゆるディープラーニングと呼ばれる技術を用いて実装されており、<a href="https://arxiv.org/abs/1611.07004">pix2pix</a> というネットワーク構造をベースにしています。実装は<a href="https://github.com/pfnet-research/chainer-pix2pix">chainer-pix2pix</a>を改造して制作しました。</p>
        <p><a href="https://razor-edge.work/material/fsmchcv/">こちら</a>で配布されている First Seed Material 様の素材（高解像度版）約7000枚を用いて学習しています。</p>
    </div>
);
