## 実行方法
``` 
cd <docker-compose.xmlがあるディレクトリ>

docker-compose build

docker network create xxxxnetwork

docker-compose -p <プロジェクト名> up -d
```

## ボリューム削除して終了

```
docker-compose -p <プロジェクト名> down -v
```

## 接続

```
docker-compose exec コンテナ名 bash
```

## 接続（VSCODE）
1. 拡張機能 Remote-Container をインストール。
2. VSCode左下の緑色のボタンを押す。
3. Remote-container:Attach to Running container を選択して、該当のコンテナを選ぶ。

## Tips
GPUが無いと無視しても大丈夫という警告
```
# python
>>> import tensorflow
2021-02-10 05:03:20.152401: W tensorflow/stream_executor/platform/default/dso_loader.cc:60] Could not load dynamic library 'libcudart.so.11.0'; dlerror: libcudart.so.11.0: cannot open shared object file: No such file or directory
2021-02-10 05:03:20.152446: I tensorflow/stream_executor/cuda/cudart_stub.cc:29] Ignore above cudart dlerror if you do not have a GPU set up on your machine.
```
attatch 後  notebookが正常に表示されない場合は、vscodeのpython拡張などを確認する。
