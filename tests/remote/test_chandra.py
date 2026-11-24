"""Backend regressions that run without local models or a Chandra server."""

import os
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from PIL import Image

from marker.remote.chandra import (
    ChandraClient,
    ChandraLayoutPredictor,
    ChandraRecognitionPredictor,
)
from marker.remote.ocr_error import HeuristicOCRErrorPredictor
from marker.remote.router import PageRouter, RouteDecision, route_page
from marker.remote.schema import LayoutBox, LayoutResult
from marker.remote.util import detect_repeat_loop
from marker.settings import settings


@pytest.fixture
def chandra_backend(monkeypatch):
    monkeypatch.setattr(settings, "MODEL_BACKEND", "chandra")


def test_pipeline_imports_without_surya_or_torch():
    script = """
import importlib.abc
import sys
class BlockLocalModels(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'surya', 'transformers'}:
            raise AssertionError(f'Unexpected local model import: {fullname}')
sys.meta_path.insert(0, BlockLocalModels())
from marker.converters.pdf import PdfConverter
from marker.models import create_model_dict
from marker.remote.chandra import ChandraLayoutPredictor
models = create_model_dict()
assert isinstance(models['layout_model'], ChandraLayoutPredictor)
assert models['inference_manager'] is None
assert models['fast_layout_model'] is None
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "MODEL_BACKEND": "chandra"},
        check=True,
        capture_output=True,
        text=True,
    )


def test_chandra_models_share_client_and_optionally_route(chandra_backend, monkeypatch):
    from marker.models import create_model_dict, shutdown_models

    monkeypatch.setattr(settings, "CHANDRA_USE_PAGE_ROUTER", True)
    models = create_model_dict(load_indra=True, no_server=False)
    assert models["layout_model"].client is models["recognition_model"].client
    assert isinstance(models["page_router"], PageRouter)
    assert "indra_model" not in models
    shutdown_models(models)


def test_layout_uses_remote_predictor_without_indra(chandra_backend):
    from marker.builders.layout import LayoutBuilder

    image = Image.new("RGB", (100, 100))
    model = Mock(return_value=[LayoutResult(bboxes=[], image_bbox=[0, 0, 100, 100])])
    builder = LayoutBuilder(model, config={"mode": "balanced"})
    assert builder.indra_model is None
    assert not builder.use_fast_layout()
    results = builder.surya_layout([SimpleNamespace(get_image=lambda **_: image)], None)
    assert len(results) == 1
    model.assert_called_once_with([image])


def test_converter_defaults_and_router_cache(chandra_backend, monkeypatch):
    from marker.converters.pdf import PdfConverter

    monkeypatch.setattr("marker.converters.download_font", lambda: None)
    monkeypatch.setattr(settings, "CHANDRA_USE_PAGE_ROUTER", True)
    converter = PdfConverter(
        {}, processor_list=[], config={"cache_pdftext_pages": False}
    )
    assert converter.mode == "balanced"
    assert converter.config["cache_pdftext_pages"] is True
    for mode in ("fast", "indra", "no_ocr", "vllm"):
        with pytest.raises(ValueError, match="only supports"):
            PdfConverter({}, processor_list=[], config={"mode": mode})
    with pytest.raises(ValueError, match="ocr_backend"):
        PdfConverter({}, processor_list=[], config={"ocr_backend": "indra"})


def test_line_builder_keeps_positional_config_and_injects_router(chandra_backend):
    from marker.builders.line import LineBuilder

    router = Mock()
    router.decide.side_effect = [
        RouteDecision("pdftext", []),
        RouteDecision("chandra", ["table"]),
    ]
    builder = LineBuilder(None, {"mode": "balanced"}, page_router=router)
    provider = SimpleNamespace(
        filepath="example.pdf", raw_pdftext_pages={3: {"page": 3}}
    )
    document = SimpleNamespace(
        pages=[SimpleNamespace(page_id=3), SimpleNamespace(page_id=7)]
    )
    assert builder.route_pages(document, provider) == (["good", "bad"], [0.0, 1.0])
    assert router.decide.call_args_list[0].args == ("example.pdf", 3, {"page": 3})
    assert router.decide.call_args_list[1].args == ("example.pdf", 7, None)


def test_heuristic_predictor_handles_multilingual_and_garbled_text():
    result = HeuristicOCRErrorPredictor()(
        [
            "This is ordinary readable text with many words.",
            "هذه فقرة عربية واضحة تحتوي على معلومات مفيدة للقارئ",
            "\ufffd" * 40,
            "!@#$%" * 10,
        ]
    )
    assert result.labels == ["good", "good", "bad", "bad"]


def test_line_builder_uses_injected_heuristic(chandra_backend):
    from marker.builders.line import LineBuilder

    builder = LineBuilder(HeuristicOCRErrorPredictor())
    lines = {0: [SimpleNamespace(spans=[SimpleNamespace(text="!@#$%" * 10)])]}
    result = builder.ocr_error_detection([SimpleNamespace(page_id=0)], lines)
    assert result.labels == ["bad"]


def test_router_handles_clean_and_missing_text():
    page = {
        "page": 0,
        "width": 600,
        "height": 800,
        "blocks": [
            {
                "bbox": [50, 100, 550, 120],
                "lines": [
                    {
                        "bbox": [50, 100, 550, 120],
                        "spans": [
                            {
                                "text": "This is a clean paragraph with ordinary readable words."
                            }
                        ],
                    }
                ],
            }
        ],
    }
    assert route_page(page).route == "pdftext"
    assert PageRouter().decide("unused.pdf", 0, None).route == "chandra"
    page["blocks"][0]["lines"][0]["spans"][0]["text"] = ""
    assert route_page(page).reasons == ["near_empty_text_layer"]


def test_remote_layout_and_full_page_ocr_parse_real_chandra_format():
    raw = '<div data-label="Text" data-bbox="100,100,900,300"><p>Recognized text</p></div>'
    client = Mock()
    client.generate.return_value = raw
    image = Image.new("RGB", (200, 100))
    layout = ChandraLayoutPredictor(client)([image])[0]
    assert layout.bboxes[0].label == "Text"
    assert layout.bboxes[0].bbox == [20, 10, 180, 30]
    result = ChandraRecognitionPredictor(client)([image], full_page=True)[0]
    assert "Recognized text" in result.blocks[0].html


def test_failed_full_page_falls_back_to_block_ocr():
    client = Mock()
    client.generate.side_effect = [
        None,
        '<div data-label="Text"><p>Recovered</p></div>',
    ]
    image = Image.new("RGB", (100, 100))
    layout = LayoutResult(
        image_bbox=[0, 0, 100, 100],
        bboxes=[
            LayoutBox(
                polygon=[[10, 10], [90, 10], [90, 50], [10, 50]],
                label="Text",
                raw_label="Text",
                position=0,
            )
        ],
    )
    result = ChandraRecognitionPredictor(client)([image], [layout], full_page=True)[0]
    assert result.blocks[0].html == "<p>Recovered</p>"
    assert [call.args[1] for call in client.generate.call_args_list] == [
        "ocr_layout",
        "ocr",
    ]


def test_client_retries_repetition_with_hotter_decoding(monkeypatch):
    monkeypatch.setattr(settings, "CHANDRA_MAX_RETRIES", 1)
    client = ChandraClient()
    create = Mock(
        side_effect=[
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="loop " * 100))
                ]
            ),
            SimpleNamespace(
                choices=[
                    SimpleNamespace(message=SimpleNamespace(content="<p>Recovered</p>"))
                ]
            ),
        ]
    )
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    assert client.generate(Image.new("RGB", (100, 100)), "ocr") == "<p>Recovered</p>"
    assert create.call_count == 2
    assert create.call_args_list[0].kwargs["temperature"] == 0.0
    assert create.call_args_list[1].kwargs["temperature"] > 0


def test_repeat_loop_detection():
    assert detect_repeat_loop("repeated words " * 100)
    assert not detect_repeat_loop("A normal sentence with distinct words.")


def test_image_conversion_through_chandra_pipeline(
    chandra_backend, monkeypatch, tmp_path
):
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict

    monkeypatch.setattr("marker.converters.download_font", lambda: None)
    raw = '<div data-label="Text" data-bbox="100,100,900,400"><p>Hello from Chandra.</p></div>'
    monkeypatch.setattr(ChandraClient, "generate", lambda self, image, prompt_type: raw)
    path = tmp_path / "scan.png"
    Image.new("RGB", (400, 500), "white").save(path)
    converter = PdfConverter(create_model_dict(), config={"disable_tqdm": True})
    output = converter(str(path))
    assert "Hello from Chandra." in output.markdown
    assert converter.page_count == 1


def test_batch_conversion_does_not_spawn_surya(chandra_backend, monkeypatch, tmp_path):
    from click.testing import CliRunner
    from marker.scripts.convert import convert_cli

    monkeypatch.setattr("marker.converters.download_font", lambda: None)
    raw = '<div data-label="Text" data-bbox="100,100,900,400"><p>Batch Chandra result.</p></div>'
    monkeypatch.setattr(ChandraClient, "generate", lambda self, image, prompt_type: raw)
    path = tmp_path / "scan.png"
    Image.new("RGB", (400, 500), "white").save(path)
    output_dir = tmp_path / "output"
    result = CliRunner().invoke(
        convert_cli,
        [
            str(path),
            "--output_dir",
            str(output_dir),
            "--disable_multiprocessing",
        ],
    )
    assert result.exit_code == 0, result.output
    markdown_files = list(output_dir.rglob("*.md"))
    assert len(markdown_files) == 1
    assert "Batch Chandra result." in markdown_files[0].read_text()


def test_surya_models_and_modes_still_initialize():
    import importlib.util

    if importlib.util.find_spec("surya") is None:
        pytest.skip("Install the optional surya extra to test its backend")
    script = """
from unittest.mock import Mock, patch
from marker.settings import settings
from marker.models import create_model_dict
from marker.ocr import IndraRecognitionPredictor
from marker.converters.pdf import PdfConverter
from marker.builders.layout import LayoutBuilder
from marker.remote.compat import LayoutBox
from surya.layout.schema import LayoutBox as SuryaLayoutBox
assert LayoutBox is SuryaLayoutBox
manager = Mock()
with patch('surya.inference.SuryaInferenceManager', return_value=manager), \\
     patch('surya.recognition.RecognitionPredictor', return_value=Mock()), \\
     patch('marker.models._ensure_surya_cache_writable'):
    models = create_model_dict()
assert models['inference_manager'] is manager
assert isinstance(models['indra_model'], IndraRecognitionPredictor)
assert models['layout_model'] is models['fast_layout_model'] is models['indra_model']
with patch('marker.converters.download_font'):
    for mode in ('fast', 'indra', 'no_ocr', 'balanced', 'vllm'):
        converter = PdfConverter(models, processor_list=[], config={'mode': mode})
        builder = converter.resolve_dependencies(LayoutBuilder)
        assert converter.mode == mode
        assert builder.indra_model is models['indra_model']
"""
    subprocess.run(
        [sys.executable, "-c", script],
        env={**os.environ, "MODEL_BACKEND": "surya"},
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("use_heuristic", [False, True])
def test_garbled_block_promotes_page_to_ocr(use_heuristic):
    from marker.builders.line import LineBuilder
    from marker.schema import BlockTypes
    from marker.schema.polygon import PolygonBox

    text = "\ufffd" * 300
    block = SimpleNamespace(
        id="text-block",
        block_type=BlockTypes.Text,
        raw_text=lambda document: text,
        text_extraction_method="pdftext",
    )
    page = SimpleNamespace(
        page_id=0,
        text_extraction_method="pdftext",
        get_image=lambda: Image.new("RGB", (100, 100)),
        polygon=PolygonBox.from_bbox([0, 0, 100, 100]),
        structure_blocks=lambda document: [block],
    )
    predictor = HeuristicOCRErrorPredictor() if use_heuristic else None
    LineBuilder(predictor).flag_bad_blocks(SimpleNamespace(pages=[page]))
    assert block.text_extraction_method == "surya"
    assert page.text_extraction_method == "surya"
