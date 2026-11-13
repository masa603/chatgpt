"""Finish editable Word equation layout and record equation identifiers."""

import json
from pathlib import Path
import re
from xml.etree import ElementTree as ET
from zipfile import ZipFile

W = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
M = 'http://schemas.openxmlformats.org/officeDocument/2006/math'
NS = {'w': W, 'm': M}


def equation_records(markdown):
    identifiers = re.findall(r'^::: \{#(eq:[\w-]+) \.numbered-equation\}', markdown, re.M)
    assert len(identifiers) == len(set(identifiers))
    references = re.findall(r'\]\(#(eq:[\w-]+)\)', markdown)
    assert set(references).issubset(identifiers)
    return [{'number': i, 'id': name, 'text_reference_count': references.count(name)}
            for i, name in enumerate(identifiers, 1)]


def finish_docx(path, expected_equations):
    path = Path(path)
    with ZipFile(path) as archive:
        infos = archive.infolist()
        entries = {info.filename: archive.read(info.filename) for info in infos}
    # Keep namespace prefixes used by mc:Ignorable in the original Word document.
    namespaces = re.findall(rb'xmlns:([\w]+)="([^"]+)"', entries['word/document.xml'])
    for prefix, uri in namespaces:
        ET.register_namespace(prefix.decode(), uri.decode())
    root = ET.fromstring(entries['word/document.xml'])
    section = root.find('.//w:sectPr', NS)
    page_width = 11906
    text_width = 9412
    if section is not None:
        size, margins = section.find('w:pgSz', NS), section.find('w:pgMar', NS)
        if size is not None and margins is not None:
            page_width = int(size.get(f'{{{W}}}w', page_width))
            text_width = page_width-int(margins.get(f'{{{W}}}left', 1247))-int(margins.get(f'{{{W}}}right', 1247))
    numbers = []
    for table in root.findall('.//w:tbl', NS):
        cells = table.findall('w:tr/w:tc', NS)
        if len(cells) != 3 or cells[1].find('.//m:oMath', NS) is None:
            continue
        number = ''.join(cells[2].itertext()).strip()
        if not re.fullmatch(r'\(\d+\)', number):
            continue
        numbers.append(int(number[1:-1]))
        props = table.find('w:tblPr', NS)
        if props is None:
            props = ET.Element(f'{{{W}}}tblPr'); table.insert(0, props)
        style = props.find('w:tblStyle', NS)
        if style is not None:
            props.remove(style)
        width = props.find('w:tblW', NS)
        if width is None:
            width = ET.SubElement(props, f'{{{W}}}tblW')
        width.set(f'{{{W}}}type', 'dxa'); width.set(f'{{{W}}}w', str(text_width))
        widths = [text_width//20, text_width-2*(text_width//20), text_width//20]
        for column, value in zip(table.findall('w:tblGrid/w:gridCol', NS), widths):
            column.set(f'{{{W}}}w', str(value))
        borders = props.find('w:tblBorders', NS)
        if borders is None:
            borders = ET.SubElement(props, f'{{{W}}}tblBorders')
        for edge in ['top', 'left', 'bottom', 'right', 'insideH', 'insideV']:
            item = borders.find('w:' + edge, NS)
            if item is None:
                item = ET.SubElement(borders, f'{{{W}}}{edge}')
            item.set(f'{{{W}}}val', 'nil')
        for cell, value in zip(cells, widths):
            cell_props = cell.find('w:tcPr', NS)
            if cell_props is None:
                cell_props = ET.Element(f'{{{W}}}tcPr'); cell.insert(0, cell_props)
            cell_width = ET.SubElement(cell_props, f'{{{W}}}tcW')
            cell_width.set(f'{{{W}}}type', 'dxa'); cell_width.set(f'{{{W}}}w', str(value))
            align = ET.SubElement(cell_props, f'{{{W}}}vAlign')
            align.set(f'{{{W}}}val', 'center')
        for row in table.findall('w:tr', NS):
            row_props = row.find('w:trPr', NS)
            if row_props is None:
                row_props = ET.Element(f'{{{W}}}trPr'); row.insert(0, row_props)
            ET.SubElement(row_props, f'{{{W}}}cantSplit')
    assert numbers == list(range(1, expected_equations + 1)), numbers
    if section is not None:
        if section.find('w:titlePg', NS) is None:
            ET.SubElement(section, f'{{{W}}}titlePg')
        page_number = section.find('w:pgNumType', NS)
        if page_number is None:
            page_number = ET.SubElement(section, f'{{{W}}}pgNumType')
        page_number.set(f'{{{W}}}start', '0')
    serialized = ET.tostring(root, encoding='utf-8', xml_declaration=True)
    root_tag = serialized.split(b'<w:document', 1)[1].split(b'>', 1)[0]
    missing = [b' xmlns:' + prefix + b'="' + uri + b'"' for prefix, uri in namespaces
               if b'xmlns:' + prefix + b'=' not in root_tag]
    serialized = serialized.replace(b'<w:document', b'<w:document' + b''.join(missing), 1)
    entries['word/document.xml'] = serialized
    temporary = path.with_suffix('.tmp.docx')
    with ZipFile(temporary, 'w') as archive:
        for info in infos:
            archive.writestr(info, entries[info.filename])
    temporary.replace(path)


def write_equation_record(path, markdown):
    records = equation_records(markdown)
    Path(path).write_text(json.dumps({'count': len(records), 'equations': records}, ensure_ascii=False, indent=2) + '\n')
    return len(records)
