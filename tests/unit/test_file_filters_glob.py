import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestFileFiltersGlob(unittest.TestCase):
    def setUp(self) -> None:
        project_root = Path(__file__).parent.parent.parent
        sys.path.insert(0, str(project_root))

    def test_excluded_files_glob_md_excludes_markdown(self):
        from api.data_pipeline import read_all_documents

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _write_text(tmp_path / "README.md", "# hello\n")
            _write_text(tmp_path / "notes.txt", "hello\n")
            _write_text(tmp_path / "main.py", "print('hi')\n")

            docs = read_all_documents(
                str(tmp_path),
                embedder_type="openai",
                excluded_files=["*.md"],
            )

            paths = {d.meta_data.get("file_path") for d in docs}
            self.assertNotIn("README.md", paths)
            self.assertIn("notes.txt", paths)
            self.assertIn("main.py", paths)

    def test_included_files_glob_md_includes_only_markdown(self):
        from api.data_pipeline import read_all_documents

        with TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _write_text(tmp_path / "README.md", "# hello\n")
            _write_text(tmp_path / "notes.txt", "hello\n")
            _write_text(tmp_path / "main.py", "print('hi')\n")

            docs = read_all_documents(
                str(tmp_path),
                embedder_type="openai",
                included_files=["*.md"],
            )

            paths = {d.meta_data.get("file_path") for d in docs}
            self.assertEqual(paths, {"README.md"})


if __name__ == "__main__":
    unittest.main()
