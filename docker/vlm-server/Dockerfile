FROM lmsysorg/sglang:latest

RUN python3 -m pip install -U 'mineru[vlm]' -i https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple --break-system-packages

ENV MINERU_MODEL_SOURCE=local

RUN cat <<'EOF' > "/root/mineru.json"
{
    "models-dir": {
        "vlm": "/root/.cache/modelscope/hub/MinerU2.0-2505-0.9B"
    },
    "config_version": "1.3.0"
}
EOF

ENTRYPOINT ["mineru-sglang-server"]
