# 実行方法
cd <docker-compose.xmlがあるディレクトリ>

docker-compose build

docker network create xxxxnetwork

docker-compose -p <プロジェクト名> up -d

# ボリューム削除して終了
docker-compose down -v

# 接続
docker-compose exec コンテナ名 bash