import os
import re
import types
from pathlib import Path

import jinja2
from fastapi.templating import Jinja2Templates
from gradio import routes
from gradio.themes.utils.fonts import GoogleFont


def _patch_env() -> None:
    os.environ["GRADIO_ANALYTICS_ENABLED"] = "False"


def _patch_font(resources_path: Path) -> None:
    # Load css font from local file when server started
    # If you want to provide it as an external url, please ensure url started with http:// or https://
    # Otherwise external CSS fonts will not be loaded
    def _patched_stylesheet(self: GoogleFont) -> dict:
        file_path = resources_path.joinpath("css").joinpath(f"{self.name}.css")
        if file_path.is_file():
            return {
                "url": None,
                "css": file_path.read_text(),
            }
        else:
            raise FileNotFoundError(f"Font css file '{file_path}' not found")

    GoogleFont.stylesheet = _patched_stylesheet


def _patch_templates(url_prefix: str) -> None:
    remove_regex: list[re.Pattern] = [
        re.compile(
            r"<meta\b[^>]*?property=[\"']og:[^\"'>]*[\"'][^>]*?>",
            flags=re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"<meta\b[^>]*?name=[\"']twitter:[^\"'>]*[\"'][^>]*?>",
            flags=re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"<link\b[^>]*?href=[\"']?[^\"'>]*fonts.googleapis.com[^\"'>]*[\"']?[^>]*?>",
            flags=re.IGNORECASE | re.DOTALL,
        ),
        re.compile(
            r"<link\b[^>]*?href=[\"']?[^\"'>]*fonts.gstatic.com[^\"'>]*[\"']?[^>]*?>",
            flags=re.IGNORECASE | re.DOTALL,
        ),
    ]
    cdn_regex: list[re.Pattern] = [re.compile(r"https?://cdnjs.cloudflare.com.*?")]

    def _do_patch(html: str) -> str:
        for pattern in remove_regex:
            html = re.sub(pattern, "", html)
        for pattern in cdn_regex:
            html = re.sub(pattern, url_prefix, html)
        return html

    def _patched_render(self: jinja2.Template, *args, **kwargs) -> str:
        html = jinja2.Template.render(self, *args, **kwargs)
        return _do_patch(html)

    async def _patched_render_async(self: jinja2.Template, *args, **kwargs) -> str:
        html = await jinja2.Template.render_async(self, *args, **kwargs)
        return _do_patch(html)

    class PatchedJinja2Templates(Jinja2Templates):
        def get_template(self, name: str) -> jinja2.Template:
            template: jinja2.Template = super().get_template(name)
            template.render = types.MethodType(_patched_render, template)
            template.render_async = types.MethodType(_patched_render_async, template)
            return template

    routes.templates = PatchedJinja2Templates(directory=routes.STATIC_TEMPLATE_LIB)
    # noinspection SpellCheckingInspection
    routes.templates.env.filters["toorjson"] = routes.toorjson


def patch(resources_path: Path, url_prefix: str) -> None:
    _patch_env()
    _patch_font(resources_path)
    _patch_templates(url_prefix)
