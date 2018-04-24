import * as React from "react";

export const About = (props: {}) => (
    <div>
        <h2>利用規約</h2>
        <ul>
            <li>変換された画像については、下記、学習データすべての利用規約について、それぞれ個別に従うものとします</li>
            <li>学習済みモデルは予告なく変更され変換結果が変わる可能性があります</li>
            <li>本アプリケーションを利用して生じた如何なる損害についても責任は負いません</li>
        </ul>        
        <h2>このアプリケーションについて</h2>
        <p>ドット絵に特化した拡大ツールです。</p>
        <p><a href="https://en.wikipedia.org/wiki/Pixel-art_scaling_algorithms" target="_blank">既存の手法</a>と比較してより自然な拡大が可能です。</p>
        <p>いわゆるディープラーニングと呼ばれる技術を用いて実装されており、<a href="https://arxiv.org/abs/1611.07004" target="_blank">pix2pix</a> というネットワーク構造をベースにしています。実装は<a href="https://github.com/pfnet-research/chainer-pix2pix" target="_blank">chainer-pix2pix</a>を改造して制作しました。</p>
        <h2>学習データ</h2>        
        <ul>
            <li><a href="https://razor-edge.work/material/fsmchcv/" target="_blank">カミソリエッジ</a>様が配布されている <a href="http://catnest0523.blog.fc2.com/blog-entry-436.html" target="_blank">First Seed Material(サイト閉鎖のため規約のアーカイブページ)</a>の素材（高解像度版）のカラーバリエーション約7000枚</li>
            <li><a href="https://mplus-fonts.osdn.jp/" target="_blank">M+フォント</a>全種から、light, thin を除いたもの</li>
            <li><a href="https://comshou.wixsite.com/com-sho/about" target="_blank">コミュ将</a>様の配布されている<a href="https://comshou.wixsite.com/com-sho/single-post/2017/04/19/RTP%E4%B8%8D%E4%BD%BF%E7%94%A8%E7%B4%A0%E6%9D%90%E3%81%BE%E3%81%A8%E3%82%81" target="_blank">タイルセット（RTP不使用版）</a></li>
        </ul>
    </div>
);
