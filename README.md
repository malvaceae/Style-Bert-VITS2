# Style-Bert-VITS2 with Amazon SageMaker

[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)を用いたモデルのトレーニングと推論を[Amazon SageMaker](https://aws.amazon.com/sagemaker/)で実行できるようにしたもの。

このリポジトリは、[Style-Bert-VITS2](https://github.com/litagin02/Style-Bert-VITS2)、[Bert-VITS2](https://github.com/fishaudio/Bert-VITS2) v2.1、および Japanese-Extra をベースにしています。  
オリジナルの作成者様に感謝します。

## 要件

以下がインストールされていること。

- AWS CLI
- Docker

## 事前準備

### Dockerイメージ

トレーニングと推論に用いるDockerイメージをECRリポジトリにプッシュする。

```sh
# ECRリポジトリ
ECR_REPOSITORY=style-bert-vits2

# アカウントID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# リージョン
AWS_REGION="${AWS_DEFAULT_REGION:-$(aws configure get region)}"

# ECRレジストリ
ECR_REGISTRY="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com"

# Dockerイメージ名
IMAGE_NAME="${ECR_REGISTRY}/${ECR_REPOSITORY}:latest"

# ECRリポジトリを作成
aws ecr create-repository --repository-name "${ECR_REPOSITORY}"

# ECRレジストリにログイン
aws ecr get-login-password | docker login --username AWS --password-stdin "${ECR_REGISTRY}"

# Dockerイメージをビルド (AMD64用イメージとしてビルドする必要あり)
docker build --platform linux/amd64 -t "${IMAGE_NAME}" .

# DockerイメージをECRリポジトリにプッシュ
docker push "${IMAGE_NAME}"
```

### 実行ロールとS3バケット

SageMakerの実行ロールと、学習データやモデルを格納するためのS3バケットを作成する。  
https://docs.aws.amazon.com/sagemaker/latest/dg/sagemaker-roles.html

## トレーニング

学習に用いる音声ファイルを以下のような構造でS3バケットに配置する。

```
inputs
└── model_name
    ├── ****.wav
    ├── ****.wav
    ├── ...
```

以下のコマンドを実行する。

```sh
# トレーニングジョブ名
TRAINING_JOB_NAME=style-bert-vits2-$(date '+%Y-%m-%d-%H-%M-%S')

# 事前準備したDockerイメージ名
IMAGE_NAME=<Dockerイメージ名>

# 実行ロールのARN
EXECUTION_ROLE_ARN=<実行ロールのARN>

# 音声ファイルを配置したパスを指すS3 URI
S3_INPUT_URI=s3://<S3バケット名>/inputs

# 成果物を格納するパスを指すS3 URI
S3_OUTPUT_URI=s3://<S3バケット名>/outputs

# インスタンスタイプ
INSTANCE_TYPE=ml.g4dn.xlarge

# 追加のストレージボリューム (GB)
VOLUME_SIZE_IN_GB=30

# 最大実行時間 (秒)
MAX_RUNTIME_IN_SECONDS=86400

# JP-Extraを使うかどうか
USE_JP_EXTRA=True

# 学習のバッチサイズ
BATCH_SIZE=4

# 学習のエポック数
EPOCHS=100

# 保存頻度 (ステップ数)
SAVE_EVERY_STEPS=1000

# 音声ファイルの音量を正規化するかどうか
NORMALIZE=False

# 音声ファイルの開始・終了にある無音区間を削除するかどうか
TRIM=False

# トレーニングジョブを実行
aws sagemaker create-training-job \
  --training-job-name "${TRAINING_JOB_NAME}" \
  --hyper-parameters "use_jp_extra=${USE_JP_EXTRA},
    batch_size=${BATCH_SIZE},
    epochs=${EPOCHS},
    save_every_steps=${SAVE_EVERY_STEPS},
    normalize=${NORMALIZE},
    trim=${TRIM}" \
  --role-arn "${EXECUTION_ROLE_ARN}" \
  --algorithm-specification "TrainingImage=${IMAGE_NAME},TrainingInputMode=File" \
  --input-data-config "ChannelName=train,
    DataSource={
      S3DataSource={
        S3DataType=S3Prefix,
        S3Uri=${S3_INPUT_URI},
        S3DataDistributionType=FullyReplicated,
      },
    }" \
  --output-data-config "S3OutputPath=${S3_OUTPUT_URI}" \
  --resource-config "InstanceType=${INSTANCE_TYPE},
    InstanceCount=1,
    VolumeSizeInGB=${VOLUME_SIZE_IN_GB}" \
  --stopping-condition "MaxRuntimeInSeconds=${MAX_RUNTIME_IN_SECONDS}"

# トレーニングジョブの完了を待機
aws sagemaker wait training-job-completed-or-stopped \
  --training-job-name "${TRAINING_JOB_NAME}"
```

学習が完了すると、`${S3_OUTPUT_URI}/${TRAINING_JOB_NAME}/output/model.tar.gz` にモデルファイルが保存される。

## 推論

以下の構造でtar.gz圧縮したモデルファイルをS3バケットに配置する。

```
model.tar.gz
└── model_name
    ├── config.json
    ├── model_name.safetensors
    └── style_vectors.npy
```

以下のコマンドを実行する。

```sh
# モデル名
MODEL_NAME=style-bert-vits2-model-$(date '+%Y-%m-%d-%H-%M-%S')

# エンドポイント設定名
ENDPOINT_CONFIG_NAME=style-bert-vits2-endpoint-config-$(date '+%Y-%m-%d-%H-%M-%S')

# エンドポイント名
ENDPOINT_NAME=style-bert-vits2-endpoint-$(date '+%Y-%m-%d-%H-%M-%S')

# 事前準備したDockerイメージ名
IMAGE_NAME=<Dockerイメージ名>

# 実行ロールのARN
EXECUTION_ROLE_ARN=<実行ロールのARN>

# モデルファイルのパスを指すS3 URI
S3_MODEL_URI=s3://<S3バケット名>/<モデルファイルのパス>

# インスタンスタイプ
INSTANCE_TYPE=ml.g4dn.xlarge

# モデルを作成
aws sagemaker create-model \
  --model-name "${MODEL_NAME}" \
  --primary-container "Image=${IMAGE_NAME},ModelDataUrl=${S3_MODEL_URI}" \
  --execution-role-arn "${EXECUTION_ROLE_ARN}"

# エンドポイント設定を作成
aws sagemaker create-endpoint-config \
  --endpoint-config-name "${ENDPOINT_CONFIG_NAME}" \
  --production-variants "VariantName=AllTraffic,
    ModelName=${MODEL_NAME},
    InitialInstanceCount=1,
    InstanceType=${INSTANCE_TYPE}"

# エンドポイントを作成
aws sagemaker create-endpoint \
  --endpoint-name "${ENDPOINT_NAME}" \
  --endpoint-config-name "${ENDPOINT_CONFIG_NAME}"

# エンドポイントの作成を待機
aws sagemaker wait endpoint-in-service \
  --endpoint-name "${ENDPOINT_NAME}"
```

エンドポイントの作成が完了すると、SageMakerの [InvokeEndpoint API](https://docs.aws.amazon.com/sagemaker/latest/APIReference/API_runtime_InvokeEndpoint.html) で音声合成が可能となる。

以下はAWS CLIで実行する例。`${AUDIO_FILE_NAME}` に結果が保存される。

```sh
# セリフ
TEXT=こんにちは、初めまして。あなたの名前はなんていうの？

# 保存ファイル名
AUDIO_FILE_NAME=test.wav

# エンドポイントを呼び出す
aws sagemaker-runtime invoke-endpoint \
  --cli-binary-format raw-in-base64-out \
  --endpoint-name "${ENDPOINT_NAME}" \
  --body "{\"text\":\"${TEXT}\"}" \
  --content-type application/json \
  --accept audio/wav \
  "${AUDIO_FILE_NAME}"
```
