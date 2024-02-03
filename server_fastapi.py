"""
API server for TTS
"""
import argparse
import os
import sys
from io import BytesIO
from typing import Optional, Union
from urllib.parse import unquote

import torch
import uvicorn
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
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
from common.tts_model import Model, ModelHolder
from config import config

ln = config.server_config.language


def raise_validation_error(msg: str, param: str):
    logger.warning(f"Validation error: {msg}")
    raise HTTPException(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        detail=[dict(type="invalid_params", msg=msg, loc=["query", param])],
    )


class AudioResponse(Response):
    media_type = "audio/wav"


def load_models(model_holder: ModelHolder):
    model_holder.models = []
    for model_name, model_paths in model_holder.model_files_dict.items():
        model = Model(
            model_path=model_paths[0],
            config_path=os.path.join(model_holder.root_dir, model_name, "config.json"),
            style_vec_path=os.path.join(
                model_holder.root_dir, model_name, "style_vectors.npy"
            ),
            device=model_holder.device,
        )
        model.load_net_g()
        model_holder.models.append(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument(
        "--dir", "-d", type=str, help="Model directory", default=config.assets_root
    )
    args = parser.parse_args()

    if args.cpu:
        device = "cpu"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_dir = args.dir
    model_holder = ModelHolder(model_dir, device)
    if len(model_holder.model_names) == 0:
        logger.error(f"Models not found in {model_dir}.")
        sys.exit(1)

    logger.info("Loading models...")
    load_models(model_holder)
    limit = config.server_config.limit
    app = FastAPI()
    allow_origins = config.server_config.origins
    if allow_origins:
        logger.warning(
            f"CORS allow_origins={config.server_config.origins}. If you don't want, modify config.yml"
        )
        app.add_middleware(
            CORSMiddleware,
            allow_origins=config.server_config.origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.logger = logger

    @app.post("/invocations", response_class=AudioResponse)
    async def invocations(
        request: Request,
        text: str = Body(..., min_length=1, max_length=limit, description=f"セリフ"),
        encoding: str = Body(None, description="textをURLデコードする(ex, `utf-8`)"),
        model_id: int = Body(0, description="モデルID。`GET /models/info`のkeyの値を指定ください"),
        speaker_name: str = Body(
            None, description="話者名(speaker_idより優先)。esd.listの2列目の文字列を指定"
        ),
        speaker_id: int = Body(
            0, description="話者ID。model_assets>[model]>config.json内のspk2idを確認"
        ),
        sdp_ratio: float = Body(
            DEFAULT_SDP_RATIO,
            description="SDP(Stochastic Duration Predictor)/DP混合比。比率が高くなるほどトーンのばらつきが大きくなる",
        ),
        noise: float = Body(DEFAULT_NOISE, description="サンプルノイズの割合。大きくするほどランダム性が高まる"),
        noisew: float = Body(
            DEFAULT_NOISEW, description="SDPノイズ。大きくするほど発音の間隔にばらつきが出やすくなる"
        ),
        length: float = Body(
            DEFAULT_LENGTH, description="話速。基準は1で大きくするほど音声は長くなり読み上げが遅まる"
        ),
        language: Languages = Body(ln, description=f"textの言語"),
        auto_split: bool = Body(DEFAULT_LINE_SPLIT, description="改行で分けて生成"),
        split_interval: float = Body(
            DEFAULT_SPLIT_INTERVAL, description="分けた場合に挟む無音の長さ（秒）"
        ),
        assist_text: Optional[str] = Body(
            None, description="このテキストの読み上げと似た声音・感情になりやすくなる。ただし抑揚やテンポ等が犠牲になる傾向がある"
        ),
        assist_text_weight: float = Body(
            DEFAULT_ASSIST_TEXT_WEIGHT, description="assist_textの強さ"
        ),
        style: Optional[Union[int, str]] = Body(DEFAULT_STYLE, description="スタイル"),
        style_weight: float = Body(DEFAULT_STYLE_WEIGHT, description="スタイルの強さ"),
        reference_audio_path: Optional[str] = Body(None, description="スタイルを音声ファイルで行う"),
    ):
        """Infer text to speech(テキストから感情付き音声を生成する)"""
        logger.info(
            f"{request.client.host}:{request.client.port}/invocations  { unquote(str(await request.json()) )}"
        )
        if model_id >= len(model_holder.models):  # /models/refresh があるためQuery(le)で表現不可
            raise_validation_error(f"model_id={model_id} not found", "model_id")

        model = model_holder.models[model_id]
        if speaker_name is None:
            if speaker_id not in model.id2spk.keys():
                raise_validation_error(
                    f"speaker_id={speaker_id} not found", "speaker_id"
                )
        else:
            if speaker_name not in model.spk2id.keys():
                raise_validation_error(
                    f"speaker_name={speaker_name} not found", "speaker_name"
                )
            speaker_id = model.spk2id[speaker_name]
        if style not in model.style2id.keys():
            raise_validation_error(f"style={style} not found", "style")
        if encoding is not None:
            text = unquote(text, encoding=encoding)
        sr, audio = model.infer(
            text=text,
            language=language,
            sid=speaker_id,
            reference_audio_path=reference_audio_path,
            sdp_ratio=sdp_ratio,
            noise=noise,
            noisew=noisew,
            length=length,
            line_split=auto_split,
            split_interval=split_interval,
            assist_text=assist_text,
            assist_text_weight=assist_text_weight,
            use_assist_text=bool(assist_text),
            style=style,
            style_weight=style_weight,
        )
        logger.success("Audio data generated and sent successfully")
        with BytesIO() as wavContent:
            wavfile.write(wavContent, sr, audio)
            return Response(content=wavContent.getvalue(), media_type="audio/wav")

    @app.get("/ping")
    def ping():
        return Response()

    logger.info(f"server listen: http://127.0.0.1:{config.server_config.port}")
    logger.info(f"API docs: http://127.0.0.1:{config.server_config.port}/docs")
    uvicorn.run(
        app, port=config.server_config.port, host="0.0.0.0", log_level="warning"
    )
