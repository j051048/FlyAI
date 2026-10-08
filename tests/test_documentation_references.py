"""Documentation checks include all repo guides and use their actual link bases."""
import importlib.util
from pathlib import Path
import subprocess


def checker():
    path = Path(__file__).resolve().parents[1] / 'tools' / 'check_references.py'
    spec = importlib.util.spec_from_file_location('documentation_checker_test',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_source_archive_checks_sidecar_phase0_and_provenance(tmp_path,monkeypatch):
    module=checker();monkeypatch.setattr(module,'REPO_ROOT',tmp_path)
    def no_git(*args,**kwargs):raise FileNotFoundError
    monkeypatch.setattr(module.subprocess,'run',no_git)
    for folder in ('sidecar','phase0','vendor/reference','.venv','scratchpad'):
        path=tmp_path/folder;path.mkdir(parents=True)
        (path/'README.md').write_text('[broken](missing.md)',encoding='utf-8')
    errors=[];module.check_markdown_links(errors)
    assert len(errors)==3
    assert any('sidecar' in e for e in errors) and any('vendor' in e for e in errors)


def test_wrong_document_relative_link_not_masked_by_repo_root(tmp_path,monkeypatch):
    module=checker();monkeypatch.setattr(module,'REPO_ROOT',tmp_path)
    monkeypatch.setattr(module,'markdown_files',lambda:[tmp_path/'docs'/'guide.md'])
    (tmp_path/'docs').mkdir();(tmp_path/'docs'/'other.md').write_text('ok')
    (tmp_path/'docs'/'guide.md').write_text('[bad](docs/other.md)\n[good](other.md)',encoding='utf-8')
    errors=[];module.check_markdown_links(errors)
    assert len(errors)==1 and 'docs/other.md' in errors[0]


def test_code_examples_and_external_links_skipped_encoded_spaces_resolved(tmp_path,monkeypatch):
    module=checker();monkeypatch.setattr(module,'REPO_ROOT',tmp_path)
    monkeypatch.setattr(module,'markdown_files',lambda:[tmp_path/'guide.md'])
    (tmp_path/'a b.md').write_text('ok')
    (tmp_path/'guide.md').write_text('```py\n[x](missing.md)\n```\n`[x](also-missing.md)`\n[x](a%20b.md)\n[x](<a b.md>)\n[x](https://example.com)\n[x](#anchor)',encoding='utf-8')
    errors=[];module.check_markdown_links(errors)
    assert errors==[]
