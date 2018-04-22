import * as React from "react";

export const About = (props: {}) => (
    <div>
        <p>ドット絵に特化した拡大ツールです。</p>
        <p><a href="https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms">既存の手法</a>よりもシャープなエッジとやや滲んだ細部の塗りが特徴です。</p>
        <p>いわゆるディープラーニングと呼ばれる技術を用いて実装されており、<a href="https://arxiv.org/abs/1611.07004">pix2pix</a> というネットワーク構造をベースにしています。実装は<a href="https://github.com/pfnet-research/chainer-pix2pix">chainer-pix2pix</a>を改造して制作しました。</p>
        <p>
            以下の素材を用いて学習しています。
            <ul>
                <li><a href="https://razor-edge.work/material/fsmchcv/">カミソリエッジ</a>様が配布されている First Seed Material 素材（高解像度版）のカラーバリエーション約7000枚</li>
                <li><a href="https://mplus-fonts.osdn.jp/">M+フォント</a>全種から、light, thin を除いたもの</li>
                <li><a href="https://comshou.wixsite.com/com-sho/about">コミュ将</a>様の配布されている<a href="https://comshou.wixsite.com/com-sho/single-post/2017/04/19/RTP%E4%B8%8D%E4%BD%BF%E7%94%A8%E7%B4%A0%E6%9D%90%E3%81%BE%E3%81%A8%E3%82%81">タイルセット（RTP不使用版）</a></li>
            </ul>
        </p>
    </div>
);
