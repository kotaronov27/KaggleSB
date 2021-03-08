# docker for jupyter

## 目的

Docker にVSCode で接続してtensorflow を使用する環境。
（普通にpython + tensorflow ランタイムとしても使えます）
一応Jupyter notebook がVSCode で使用できるようになったみたいなので、  
その点を考慮してみたものの、VSCodeでのJupyter notebook がやや不安定、不便みたいなので、  
様子を見た方がよいかも。

## 実行方法

``` shell:
cd <docker-compose.xmlがあるディレクトリ>

docker-compose build

docker network create xxxxnetwork

#.env　で指定していない場合
docker-compose -p <プロジェクト名> up -d
```

## ボリューム削除して終了

``` shell:

#.env　で指定していない場合
docker-compose -p <プロジェクト名> down -v
```

## 接続

``` shell:
docker-compose exec tf bash
```

## 接続（VSCODE）

1. 拡張機能 Remote-Container をインストール。
2. VSCode左下の緑色のボタンを押す。
3. Remote-container:Attach to Running container を選択して、該当のコンテナを選ぶ。

## Tips

GPUが無いと無視しても大丈夫という警告

``` python:
# python
>>> import tensorflow
2021-02-10 05:03:20.152401: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-02-10 05:03:20.152446: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
```

attatch 後  notebookが正常に表示されない場合は、vscodeのpython拡張などを確認する。
