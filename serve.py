import argparse
import itertools
import sys
from io import BytesIO
from pathlib import Path
from typing import NoReturn

import torch
import uvicorn
from fastapi import (
    Body,
    FastAPI,
    HTTPException,
    Request,
    Response,
    status,
)
from scipy.io import wavfile

from common.constants import (
    DEFAULT_ASSIST_TEXT_WEIGHT,
    DEFAULT_LENGTH,
    DEFAULT_LINE_SPLIT,
    DEFAULT_NOISE,
    DEFAULT_NOISEW,
    DEFAULT_SDP_RATIO,
    DEFAULT_SPLIT_INTERVAL,
    DEFAULT_STYLE,
    DEFAULT_STYLE_WEIGHT,
    Languages,
)
from common.log import logger
from common.tts_model import Model
from config import config


class WavResponse(Response):
    media_type = "audio/wav"


def raise_validation_error(msg: str, loc: tuple[int | str, ...]) -> NoReturn:
    logger.warning(f"Validation error: msg={msg}, loc={loc}")

    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[
            {
                "type": "invalid_params",
                "loc": loc,
                "msg": msg,
            },
        ],
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model_dir", required=True)
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cpu" if not torch.cuda.is_available() else "cuda"

    def get_model(path: Path) -> Model | None:
        model_paths = sorted(
            itertools.chain(
                path.glob("*.pth"),
                path.glob("*.pt"),
                path.glob("*.safetensors"),
            ),
        )

        if not model_paths:
            logger.warning(f"No model files found in {path}, so skip it")
            return None

        return Model(
            str(model_paths[0]),
            str(path / "config.json"),
            str(path / "style_vectors.npy"),
            device,
        )

    models = [get_model(path) for path in Path(args.model_dir).iterdir() if path.is_dir()]
    models = {model.hps.model_name: model for model in models if isinstance(model, Model)}

    if not models:
        logger.error(f"No models found in {args.model_dir}")
        sys.exit(1)

    for model in models.values():
        model.load_net_g()

    app = FastAPI(openapi_url=None)

    @app.post("/invocations")
    async def invocations(
        request: Request,
        text: str = Body(
            ...,
            title="セリフ",
            min_length=1,
            max_length=config.server_config.limit,
        ),
        model_name: str = Body(
            next(iter(models)),
            title="モデル名",
        ),
        speaker_id: int = Body(
            0,
            title="話者ID",
        ),
        language: Languages = Body(
            config.server_config.language,
            title="セリフの言語",
        ),
        sdp_ratio: float = Body(
            DEFAULT_SDP_RATIO,
            title="SDP(Stochastic Duration Predictor)/DP混合比",
            description="比率が高くなるほどトーンのばらつきが出やすくなる。",
        ),
        noise: float = Body(
            DEFAULT_NOISE,
            title="サンプルノイズの割合",
            description="大きくするほどランダム性が高まる。",
        ),
        noisew: float = Body(
            DEFAULT_NOISEW,
            title="SDPノイズ",
            description="大きくするほど発音の間隔にばらつきが出やすくなる。",
        ),
        length: float = Body(
            DEFAULT_LENGTH,
            title="話速",
            description="大きくするほど読み上げが遅くなる。",
        ),
        line_split: bool = Body(
            DEFAULT_LINE_SPLIT,
            title="改行で分けて生成",
        ),
        split_interval: float = Body(
            DEFAULT_SPLIT_INTERVAL,
            title="改行ごとに挟む無音の長さ（秒）",
        ),
        assist_text: str | None = Body(
            None,
            title="Assist text",
            description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし、抑揚やテンポ等が犠牲になる傾向がある。",
        ),
        assist_text_weight: float = Body(
            DEFAULT_ASSIST_TEXT_WEIGHT,
            title="Assist textの強さ",
        ),
        style: str = Body(
            DEFAULT_STYLE,
            title="スタイル",
        ),
        style_weight: float = Body(
            DEFAULT_STYLE_WEIGHT,
            title="スタイルの強さ",
        ),
        reference_audio_path: str | None = Body(
            None,
            title="参照音声のパス",
            description="スタイルを音声ファイルで行う。",
        ),
        tone: list[int] | None = Body(
            None,
            title="アクセント調整",
        ),
    ):
        logger.info(f"/invocations {await request.json()}")

        if model_name not in models:
            raise_validation_error(
                msg=f"No {model_name} found",
                loc=("body", "model_name"),
            )

        model = models[model_name]

        if speaker_id not in model.id2spk:
            raise_validation_error(
                msg=f"No {speaker_id} found",
                loc=("body", "speaker_id"),
            )

        if style not in model.style2id:
            raise_validation_error(
                msg=f"No {style} found",
                loc=("body", "style"),
            )

        result = model.infer(
            text=text,
            language=language,
            sid=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noisew=noisew,
            length=length,
            line_split=line_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
            given_tone=tone,
        )

        logger.success("Audio data generated and sent successfully")

        with BytesIO() as wav:
            wavfile.write(wav, *result)
            wavContent = wav.getvalue()

        return WavResponse(wavContent)

    @app.get("/ping")
    async def ping():
        return Response()

    logger.info(f"Server listen: http://0.0.0.0:{config.server_config.port}")

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=config.server_config.port,
        log_level="warning",
    )
