# Hessian Filter
ヘシアンフィルタのプログラムです．そもそもアルゴリズムが難しく，実行手順も複雑なので，メモがてら書き残しておきます．

## masked.py
CT画像から，肺領域のみを取り出します．

## crop_img.py
取り出された肺領域画像はそれ以外背景と化して，計算上不要なので，できるだけ背景を取り除き，画像サイズを小さくします．

## tf_conv3d.py
2階ガウス微分フィルタによって画像フィルタリングを行います．計算を高速化するため，TensorFlowによって実行しますが，おそらく，V1であれば動くと思います．

## get_eigen_values.py
固有値を固有値分解によって生成します．このプログラムは，ものすごく時間がかかります．気長に待ちましょう．

## get_linearity.py
線状度を算出します．マルチスケールに対応しています．

## num_to_mhd.py
.npyファイルを.mhdファイルに変換します．

## get_patch.py
線状度が最大となる点をパッチの中心として，パッチ画像を取得します．