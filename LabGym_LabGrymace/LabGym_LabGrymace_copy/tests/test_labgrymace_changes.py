"""Verify the changes this build makes to upstream LabGym.

Each test checks one documented change. Most read the source as text, so they run
without TensorFlow. One functional test renders a confusion matrix and is skipped
when TensorFlow is not installed.
"""
import os
import re

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(HERE, '..')
CATEGORIZER = open(os.path.join(PKG, 'categorizer.py'), encoding='utf-8').read()
NOSPACE = CATEGORIZER.replace(' ', '')


def _method_source(name):
    """Return the source of a def by name, up to the next def at any indent."""
    m = re.search(r'\n[ \t]*def ' + re.escape(name) + r'\b', CATEGORIZER)
    assert m, 'def ' + name + ' not found'
    rest = CATEGORIZER[m.end():]
    nxt = re.search(r'\n[ \t]*def ', rest)
    return rest[:nxt.start()] if nxt else rest


def test_version_is_pinned():
    # This build is pinned to 2.9.0, the version LabGrymace was calibrated against.
    init = open(os.path.join(PKG, '__init__.py'), encoding='utf-8').read()
    assert "__version__='2.9.0'" in init.replace(' ', '')


def test_confusion_matrix_uses_enlarged_fonts():
    # The diagnostic matrix uses large title, axis, and class-label fonts; the cell
    # counts keep the base size.
    for token in ('FS_TITLE=28', 'FS_AXIS=26', 'FS_TICK=20', 'FS_CELL=15'):
        assert token in NOSPACE
    # The colorbar carries a bold vertical 'Count' label.
    assert "cbar.set_label('Count'" in NOSPACE


def test_test_categorizer_reads_every_example():
    # test_categorizer reads each file directly, so no example is dropped. It must
    # not build the batched loader, which rounds its length down.
    src = _method_source('test_categorizer')
    assert 'os.listdir' in src
    assert 'DatasetFromPath_AA(' not in src


def test_loader_length_rounds_down():
    # DatasetFromPath_AA rounds its length down. Using it for evaluation would drop
    # the last partial batch, which is why test_categorizer reads files directly.
    assert 'np.floor' in _method_source('__len__')


def test_training_batch_sizes():
    # Both training paths use the default batch schedule, whose largest batch is 32.
    # The on-the-fly variant no longer uses 128.
    onfly = _method_source('train_combnet_onfly').replace(' ', '')
    assert 'batch_size=32' in _method_source('train_combnet').replace(' ', '')
    assert 'batch_size=32' in onfly
    assert 'batch_size=128' not in onfly


def test_keras_savedmodel_loader_present():
    # A wrapper loads both .keras files and legacy SavedModel folders under Keras 3.
    assert 'def _load_model' in CATEGORIZER
    assert 'saved_model.load' in CATEGORIZER


def test_confusion_matrix_renders_with_enlarged_fonts(monkeypatch, tmp_path):
    # Render a small matrix and read the font sizes back from the live figure.
    pytest.importorskip('tensorflow')
    import sys
    import types
    torch = types.ModuleType('torch')
    cuda = types.ModuleType('torch.cuda')
    cuda.is_available = lambda: False
    torch.cuda = cuda
    sys.modules.setdefault('torch', torch)
    sys.modules.setdefault('torch.cuda', cuda)

    from LabGym_LabGrymace_copy.categorizer import Categorizers
    import matplotlib.pyplot as plt

    seen = {}
    real_savefig = plt.savefig

    def spy(*args, **kwargs):
        ax = plt.gcf().axes[0]
        seen['title'] = ax.title.get_fontsize()
        seen['xlabel'] = ax.xaxis.label.get_fontsize()
        seen['tick'] = ax.get_xticklabels()[0].get_fontsize()
        return real_savefig(*args, **kwargs)

    monkeypatch.setattr(plt, 'savefig', spy)
    y_true = [0, 0, 1, 1]
    y_pred = [0, 1, 1, 1]
    report = {'0': {'f1-score': 0.5}, '1': {'f1-score': 0.8}}
    Categorizers()._plot_diagnostic_confusion_matrix(
        y_true, y_pred, ['0', '1'], report, str(tmp_path))

    assert seen['title'] == 28
    assert seen['xlabel'] == 26
    assert seen['tick'] == 20
