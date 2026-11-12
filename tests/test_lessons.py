"""Run every ```python block of every lesson, top to bottom, so the lessons never go stale."""
import pathlib
import re
import pytest

LESSONS = sorted(pathlib.Path(__file__).parent.parent.joinpath('lessons').glob('*.md'))

@pytest.mark.parametrize('lesson', LESSONS, ids=lambda p: p.name)
def test_lesson_runs(lesson):
    blocks = re.findall(r'```python\n(.*?)```', lesson.read_text(), flags=re.S)
    namespace = {}
    for block in blocks:
        exec(compile(block, str(lesson), 'exec'), namespace)
