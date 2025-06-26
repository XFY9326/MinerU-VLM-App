import asyncio
import base64
import dataclasses
import functools
import json
import os
import re
import shutil
import time
import uuid
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

import dataclasses_json
import gradio as gr
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from gradio_pdf import PDF
from loguru import logger
from mineru.backend.vlm.base_predictor import (
    DEFAULT_MAX_NEW_TOKENS,
    DEFAULT_NO_REPEAT_NGRAM_SIZE,
    DEFAULT_PRESENCE_PENALTY,
    DEFAULT_REPETITION_PENALTY,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_TOP_P,
    DEFAULT_USER_PROMPT,
)
from mineru.backend.vlm.sglang_client_predictor import SglangClientPredictor
from mineru.backend.vlm.token_to_middle_json import result_to_middle_json
from mineru.backend.vlm.vlm_middle_json_mkcontent import MakeMode, union_make
from mineru.cli.common import convert_pdf_bytes_to_bytes_by_pypdfium2, read_fn
from mineru.data.data_reader_writer import FileBasedDataWriter
from mineru.utils.draw_bbox import draw_layout_bbox
from mineru.utils.pdf_image_tools import load_images_from_pdf

import gr_offline

# ==================== 全局配置 ====================
SERVER_URL = os.getenv("VLM_SERVER_URL", "http://127.0.0.1:30000")

# 项目路径
PROJECT_ROOT = Path(__file__).parent
RESOURCES_PATH = PROJECT_ROOT / "resources"
ASSETS_PATH = PROJECT_ROOT / "assets"
HEADER_HTML_PATH = ASSETS_PATH / "header.html"
INDEX_JS_PATH = ASSETS_PATH / "index.js"

# 文件存储
FILES_ROOT = PROJECT_ROOT / "files"
WORKSPACE_ROOT = FILES_ROOT / "workspace"
ARCHIVE_ROOT = FILES_ROOT / "archive"
WORKSPACE_ROOT.mkdir(parents=True, exist_ok=True)
ARCHIVE_ROOT.mkdir(parents=True, exist_ok=True)

# LaTeX 分隔符
LATEX_DELIMITERS = [
    {"left": "$$", "right": "$$", "display": True},
    {"left": "$", "right": "$", "display": False},
    {"left": "\\(", "right": "\\)", "display": False},
    {"left": "\\[", "right": "\\]", "display": True},
]

# Markdown 图片正则
MD_IMAGE_PATTERN = re.compile(r"!\[[^]]*]\(([^)]+)\)")


# ==================== 类型定义 ====================
@dataclasses.dataclass(frozen=True)
class PDFParsedResult:
    original_file_name: str
    output_result_dir: Path
    origin_pdf_path: Path
    layout_pdf_path: Path
    markdown_path: Path


@dataclasses.dataclass
class PageState(dataclasses_json.DataClassJsonMixin):
    job_id: str | None = None
    origin_file_name: str | None = None
    origin_file_path: Path | None = dataclasses.field(
        metadata=dataclasses_json.config(
            decoder=lambda i: Path(i) if i else None,
            encoder=lambda i: str(i) if i else None,
        ),
        default=None,
    )

    @staticmethod
    def parse(s: str | None) -> "PageState":
        return PageState.from_json(s) if s else PageState()

    def dump(self) -> str:
        return self.to_json(ensure_ascii=False)


# ==================== Predictor 单例 ====================
@functools.lru_cache(maxsize=1, typed=True)
def get_predictor() -> SglangClientPredictor:
    """获取 SglangClientPredictor 实例"""
    return SglangClientPredictor(server_url=SERVER_URL)


# ==================== PDF 解析与转换 ====================
def analyze_pdf(
    pdf_bytes: bytes,
    image_dir: Path,
    predictor: SglangClientPredictor,
    prompts: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    no_repeat_ngram_size: int,
    max_new_tokens: int,
) -> tuple[dict, list[str]]:
    """
    对 PDF 字节流进行批量预测，返回中间 JSON 及预测结果列表
    """
    images_list, pdf_doc = load_images_from_pdf(pdf_bytes)
    img_b64_list = [img_dict["img_base64"] for img_dict in images_list]
    results = predictor.batch_predict(
        images=img_b64_list,
        prompts=prompts,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_new_tokens=max_new_tokens,
    )
    writer = FileBasedDataWriter(str(image_dir))
    middle_json = result_to_middle_json(results, images_list, pdf_doc, writer)
    return middle_json, results


def parse_pdf(
    job_id: str,
    base_name: str,
    pdf_path: Path,
    prompts: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    no_repeat_ngram_size: int,
    max_new_tokens: int,
    full_export: bool,
    start_page: int = 0,
    end_page: int | None = None,
) -> PDFParsedResult:
    """
    解析 PDF，生成布局图、Markdown、可选额外输出
    """
    job_dir = WORKSPACE_ROOT / job_id
    job_dir.mkdir(parents=True, exist_ok=True)
    image_dir = job_dir / "images"

    parsed = PDFParsedResult(
        original_file_name=base_name,
        output_result_dir=job_dir,
        origin_pdf_path=pdf_path,
        layout_pdf_path=job_dir / f"{base_name}_layout.pdf",
        markdown_path=job_dir / f"{base_name}.md",
    )

    pdf_bytes = convert_pdf_bytes_to_bytes_by_pypdfium2(pdf_path, start_page, end_page)
    predictor = get_predictor()
    middle_json, infer_results = analyze_pdf(
        pdf_bytes=pdf_bytes,
        image_dir=image_dir,
        predictor=predictor,
        prompts=prompts,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_new_tokens=max_new_tokens,
    )

    # 生成 Markdown 文件
    pdf_info = middle_json["pdf_info"]
    md_str = union_make(pdf_info, MakeMode.MM_MD, image_dir.name)
    parsed.markdown_path.write_text(str(md_str), encoding="utf-8")

    # 生成额外输出
    if full_export:
        content_list = union_make(pdf_info, MakeMode.CONTENT_LIST, image_dir.name)
        (job_dir / f"{base_name}_content_list.json").write_text(
            json.dumps(content_list, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        (job_dir / f"{base_name}_middle.json").write_text(
            json.dumps(middle_json, ensure_ascii=False, indent=4),
            encoding="utf-8",
        )
        (job_dir / f"{base_name}_model_output.txt").write_text(
            "\n".join(["-" * 50] + infer_results),
            encoding="utf-8",
        )

    # 绘制布局
    draw_layout_bbox(
        pdf_info=pdf_info,
        pdf_bytes=pdf_bytes,
        out_path=parsed.layout_pdf_path.parent,
        filename=parsed.layout_pdf_path.name,
    )
    return parsed


# ==================== 工具函数 ====================


def zip_directory(src_dir: Path, dest_zip: Path) -> Path:
    """将目录压缩为 ZIP 文件"""
    dest_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(dest_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for file in src_dir.rglob("*"):
            if file.is_file():
                zf.write(file, arcname=file.relative_to(src_dir))
    return dest_zip


def embed_images_in_md(md: str, img_dir: Path) -> str:
    """将 Markdown 中的本地图片路径替换为 Base64 编码"""

    def _repl(m: re.Match):
        rel = m.group(1)
        full = img_dir / rel
        b64 = base64.b64encode(full.read_bytes()).decode("utf-8")
        return f"![{rel}](data:image/jpeg;base64,{b64})"

    return MD_IMAGE_PATTERN.sub(_repl, md)


# ==================== Gradio 接口封装 ====================


def to_markdown(
    page_state: str | None,
    prompts: str,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
    presence_penalty: float,
    no_repeat_ngram_size: int,
    max_new_tokens: int,
    full_export: bool,
    start_page: int,
    max_pages: int,
) -> tuple[str, str, str]:
    """Gradio: 转换为 Markdown 并返回预览、原始文案、布局 PDF 路径"""
    state = PageState.parse(page_state)
    if not state.job_id:
        raise ValueError("任务ID不能为空")
    if not state.origin_file_path:
        raise ValueError("未找到上传的原始文件")
    file = state.origin_file_path.resolve()

    start = time.time()
    logger.info(f"Job started: {state.job_id}")
    parsed = parse_pdf(
        job_id=state.job_id,
        base_name=state.origin_file_name or state.job_id,
        pdf_path=file,
        prompts=prompts,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        presence_penalty=presence_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        max_new_tokens=max_new_tokens,
        full_export=full_export,
        start_page=start_page - 1,
        end_page=start_page - 1 + max_pages - 1,
    )
    logger.info(f"Job finished: {state.job_id} ({time.time() - start} s)")

    md = parsed.markdown_path.read_text(encoding="utf-8")
    md_preview = embed_images_in_md(md, parsed.output_result_dir)
    return md_preview, md, str(parsed.layout_pdf_path)


def to_archive(job_id: str) -> str:
    """Gradio: 打包当前文档，返回 ZIP 路径"""
    if not job_id:
        raise ValueError("任务ID不能为空")
    output_dir = WORKSPACE_ROOT / job_id
    if not output_dir.is_dir():
        raise FileNotFoundError("输出目录不存在")
    zip_path = ARCHIVE_ROOT / f"{job_id}.zip"
    return str(zip_directory(output_dir, zip_path))


def to_pdf(
    file_path: str | None, page_state: str | None
) -> tuple[str | None, str | None, str | None]:
    """Gradio: 上传文件后初始化任务，返回任务 ID，PDF 路径 和 状态"""
    if file_path is None:
        return None, None, page_state
    src = Path(file_path).resolve()
    job_id = str(uuid.uuid4())
    dest_dir = WORKSPACE_ROOT / job_id
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{src.stem}_origin.pdf"
    if src.suffix.lower() == ".pdf":
        shutil.copyfile(src, dest)
    else:
        dest.write_bytes(read_fn(src))

    state = PageState.parse(page_state)
    state.job_id = job_id
    state.origin_file_name = src.stem
    state.origin_file_path = dest

    logger.info(f"New job created: {job_id}")

    return job_id, str(dest), state.dump()


def clear_temp(job_id: str | None, upload_file: str | None) -> None:
    """Gradio: 清理临时文件"""
    if job_id:
        workspace_path = WORKSPACE_ROOT / job_id
        archive_path = ARCHIVE_ROOT / f"{job_id}.zip"

        if workspace_path.exists():
            try:
                shutil.rmtree(workspace_path)
                logger.info(f"Removed workspace directory: {workspace_path}")
            except OSError as e:
                logger.error(f"Error removing workspace directory {workspace_path}: {e}")

        if archive_path.exists():
            try:
                archive_path.unlink()
                logger.info(f"Removed archive file: {archive_path}")
            except OSError as e:
                logger.error(f"Error removing archive file {archive_path}: {e}")

    if upload_file:
        upload_path = Path(upload_file)
        if upload_path.exists():
            try:
                # Gradio 的临时文件可能是一个目录或文件
                if upload_path.is_dir():
                    shutil.rmtree(upload_path)
                else:
                    upload_path.unlink()
                logger.info(f"Removed temporary upload file/directory: {upload_path}")
            except OSError as e:
                logger.error(f"Error removing temporary upload {upload_path}: {e}")


def clean_old_items(target_dir: Path, expire: int = 86400) -> None:
    """
    清理指定目录下最后修改时间超过指定时间的文件和文件夹。
    """
    now = time.time()
    cutoff = now - expire
    target_path = Path(target_dir)

    if not target_path.exists() or not target_path.is_dir():
        return

    for item in target_path.iterdir():
        try:
            if item.is_file() or item.is_dir():
                mtime = item.stat().st_mtime
                if mtime < cutoff:
                    if item.is_dir():
                        shutil.rmtree(item)
                    else:
                        item.unlink()
        except Exception as e:
            logger.error(f"Failed to delete {item}: {e}")


async def scheduled_cleaner(interval_seconds: int = 3600) -> None:
    """
    每隔 interval_seconds 执行一次清理任务
    """
    clean_old_items(ARCHIVE_ROOT)
    clean_old_items(WORKSPACE_ROOT)

    while True:
        await asyncio.sleep(interval_seconds)
        clean_old_items(ARCHIVE_ROOT)
        clean_old_items(WORKSPACE_ROOT)


def build_gradio_app() -> gr.Blocks:
    """构建 Gradio Web 界面"""
    app = gr.Blocks(
        title="MinerU: PDF 文档提取",
        analytics_enabled=False,
        css="footer{display:none !important}",
        js=INDEX_JS_PATH.read_text(encoding="utf-8"),
    )
    with app:
        # 状态区
        page_state = gr.State()
        # Header
        gr.HTML(HEADER_HTML_PATH.read_text(encoding="utf-8"))
        # 上传与参数区
        with gr.Row():
            with gr.Column(variant="panel", scale=5):
                upload_file = gr.File(
                    label="上传PDF或图片",
                    file_types=[".pdf", ".png", ".jpeg", ".jpg"],
                )
                input_panel = gr.Column(visible=False)
                with input_panel:
                    with gr.Row():
                        start_page = gr.Number(
                            label="起始页码", value=1, precision=0, minimum=1, step=1
                        )
                        max_pages = gr.Number(
                            label="最大处理页数",
                            value=10,
                            precision=0,
                            minimum=1,
                            maximum=20,
                        )
                    with gr.Accordion("高级参数", open=False):
                        prompts = gr.TextArea(
                            label="提示词 Prompt (不建议修改)",
                            value=DEFAULT_USER_PROMPT,
                            lines=3,
                        )
                        with gr.Row():
                            temperature = gr.Slider(
                                label="温度 (temperature)",
                                minimum=0.0,
                                maximum=2.0,
                                step=0.1,
                                value=DEFAULT_TEMPERATURE,
                            )
                            top_p = gr.Slider(
                                label="Top-p (采样概率)",
                                minimum=0.0,
                                maximum=1.0,
                                step=0.1,
                                value=DEFAULT_TOP_P,
                            )
                        with gr.Row():
                            top_k = gr.Slider(
                                label="Top-k (采样数量)",
                                minimum=0,
                                maximum=100,
                                step=1,
                                value=DEFAULT_TOP_K,
                            )
                            repetition_penalty = gr.Slider(
                                label="重复惩罚 (repetition_penalty)",
                                minimum=0.0,
                                maximum=2.0,
                                step=0.1,
                                value=DEFAULT_REPETITION_PENALTY,
                            )
                        with gr.Row():
                            presence_penalty = gr.Slider(
                                label="主题惩罚 (presence_penalty)",
                                minimum=-2.0,
                                maximum=2.0,
                                step=0.1,
                                value=DEFAULT_PRESENCE_PENALTY,
                            )
                            no_repeat_ngram_size = gr.Slider(
                                label="不重复 n-gram 大小",
                                minimum=0,
                                maximum=10,
                                step=1,
                                value=DEFAULT_NO_REPEAT_NGRAM_SIZE,
                            )
                        max_new_tokens = gr.Slider(
                            label="最大生成长度 (max_new_tokens)",
                            minimum=1024,
                            maximum=32768,
                            step=64,
                            value=DEFAULT_MAX_NEW_TOKENS,
                        )
                        with gr.Row():
                            full_export = gr.Checkbox(
                                label="导出原始数据 (Debug)", value=False
                            )
                    with gr.Row(equal_height=True):
                        convert_btn = gr.Button("开始转换", interactive=False)
                        clear_btn = gr.ClearButton(value="清空输入")
            with gr.Column(variant="panel", scale=5):
                job_name = gr.Textbox(label="任务ID", interactive=False)
                package_btn = gr.Button("打包当前文档", interactive=False)
                output_file = gr.File(
                    label="下载文档", visible=False, interactive=False
                )
        # 预览区
        with gr.Row(equal_height=True):
            with gr.Column(variant="panel", scale=5):
                pdf_preview = PDF(label="PDF 预览", interactive=False, height=1000)
            with gr.Column(variant="panel", scale=5):
                with gr.Tabs():
                    with gr.Tab("Markdown 预览"):
                        md_preview = gr.Markdown(
                            label="Markdown 预览",
                            height=1000,
                            show_copy_button=True,
                            latex_delimiters=LATEX_DELIMITERS,
                            line_breaks=True,
                        )
                    with gr.Tab("Markdown 文本"):
                        md_text = gr.TextArea(lines=45, show_copy_button=True)

        # 交互逻辑
        upload_file.change(
            fn=to_pdf,
            inputs=[upload_file, page_state],
            outputs=[job_name, pdf_preview, page_state],
        ).then(
            fn=lambda f: [gr.update(interactive=bool(f))] * 2,
            inputs=upload_file,
            outputs=[convert_btn, package_btn],
        ).then(
            fn=lambda f: gr.update(interactive=not bool(f)),
            inputs=upload_file,
            outputs=upload_file,
        ).then(
            fn=lambda f: gr.update(visible=bool(f)),
            inputs=upload_file,
            outputs=input_panel,
        )
        convert_btn.click(
            fn=lambda: [gr.update(interactive=False)] * 4,
            inputs=None,
            outputs=[upload_file, convert_btn, clear_btn, package_btn],
        ).then(
            fn=to_markdown,
            inputs=[
                page_state,
                prompts,
                temperature,
                top_p,
                top_k,
                repetition_penalty,
                presence_penalty,
                no_repeat_ngram_size,
                max_new_tokens,
                full_export,
                start_page,
                max_pages,
            ],
            outputs=[md_preview, md_text, pdf_preview],
        ).then(
            fn=lambda: [gr.update(interactive=True)] * 4,
            inputs=None,
            outputs=[upload_file, convert_btn, clear_btn, package_btn],
        )
        job_name.change(
            fn=lambda j: gr.update(interactive=bool(j)),
            inputs=job_name,
            outputs=package_btn,
        )
        package_btn.click(fn=to_archive, inputs=[job_name], outputs=output_file)
        output_file.change(
            fn=lambda f: gr.update(visible=bool(f)),
            inputs=output_file,
            outputs=output_file,
        )
        clear_btn.click(
            fn=clear_temp, inputs=[job_name, upload_file], outputs=None
        ).then(fn=lambda: 1, inputs=None, outputs=start_page).then(
            fn=lambda: 10, inputs=None, outputs=max_pages
        )
        clear_btn.add(
            [
                page_state,
                upload_file,
                pdf_preview,
                md_preview,
                md_text,
                job_name,
                output_file,
                full_export,
            ]
        )
    return app


# ==================== FastAPI 集成 ====================
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    clean_task = asyncio.create_task(scheduled_cleaner())

    yield

    clean_task.cancel()
    try:
        await clean_task
    except asyncio.CancelledError:
        pass


api_app = FastAPI(lifespan=lifespan)
api_app.mount("/res", StaticFiles(directory=RESOURCES_PATH), name="res")
gr_offline.patch(resources_path=RESOURCES_PATH, url_prefix="./res")


@api_app.get("/favicon.ico", include_in_schema=False)
async def favicon() -> FileResponse:
    return FileResponse(ASSETS_PATH / "favicon.ico")


# 挂载 Gradio
gr.mount_gradio_app(
    app=api_app,
    blocks=build_gradio_app(),
    pwa=True,
    path="",
    show_api=False,
)

# ==================== 应用入口 ====================
if __name__ == "__main__":
    uvicorn.run(
        app="main:api_app",
        host="localhost",
        port=7860,
    )
