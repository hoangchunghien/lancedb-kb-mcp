"""Split text files into chunks that remember where they came from."""

import re

MAX_CHARS = 1200


def chunk_text(text, max_chars=MAX_CHARS):
    """Yield (heading, chunk, line_start, line_end), 1-based inclusive lines.

    Markdown headings start a new chunk and are kept as the chunk's heading;
    long sections are cut at blank lines, so a paragraph is never split.
    """
    lines = text.splitlines()
    heading, buf, start = "", [], 1

    def flush(end):
        body = "\n".join(buf).strip()
        if body:
            yield heading, body, start, end

    for number, line in enumerate(lines, 1):
        is_heading = re.match(r"#{1,6}\s+\S", line)
        too_long = (
            sum(len(x) + 1 for x in buf) + len(line) > max_chars and not line.strip()
        )
        if (is_heading or too_long) and buf:
            yield from flush(number - 1)
            buf, start = [], number
        if is_heading:
            heading = line.lstrip("#").strip()
        buf.append(line)
    yield from flush(len(lines))
