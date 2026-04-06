# python /mnt/sda/HiP-AD-custom/bench2drive/tools/hc_route_excel.py -f <json들이 있는 폴더>

import argparse
import glob
import json
import os
import zipfile
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape


DEFAULT_ROUTES_FILE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "leaderboard",
        "data",
        "bench2drive220.xml",
    )
)

HEADERS = [
    "ScenarioID",
    "Town",
    "Task",
    "Success",
    "Route Completed",
]


def is_success_record(record):
    """Keep the success criterion aligned with merge_route_json.py."""
    if record.get("status") not in {"Completed", "Perfect"}:
        return False

    for infraction_name, values in record.get("infractions", {}).items():
        if infraction_name == "min_speed_infractions":
            continue
        if len(values) > 0:
            return False

    return True


def extract_scenario_id(record):
    route_id = record.get("route_id", "")
    route_parts = route_id.split("_")
    if len(route_parts) >= 2 and route_parts[1]:
        return route_parts[1]

    save_name = record.get("save_name", "")
    save_parts = save_name.split("_")
    if len(save_parts) >= 2 and save_parts[0] == "RouteScenario":
        return save_parts[1]

    return ""


def normalize_task_name(name):
    if "_" not in name:
        return name

    stem, suffix = name.rsplit("_", 1)
    return stem if suffix.isdigit() else name


def load_route_catalog(routes_file):
    root = ET.parse(routes_file).getroot()
    route_catalog = []
    seen_route_ids = set()

    for route in root.findall("route"):
        scenario_id = str(route.attrib.get("id", "")).strip()
        if not scenario_id:
            continue

        if scenario_id in seen_route_ids:
            raise ValueError(f"Duplicate route id found in {routes_file}: {scenario_id}")
        seen_route_ids.add(scenario_id)

        scenario = route.find("./scenarios/scenario")
        task = ""
        if scenario is not None:
            task = scenario.attrib.get("type") or normalize_task_name(
                scenario.attrib.get("name", "")
            )

        route_catalog.append(
            {
                "ScenarioID": scenario_id,
                "Town": route.attrib.get("town", ""),
                "Task": task,
            }
        )

    if not route_catalog:
        raise ValueError(f"No routes found in {routes_file}")

    return route_catalog


def record_preference_key(record):
    return (
        str(record.get("save_name", "")),
        float(record.get("scores", {}).get("score_route", 0) or 0),
        1 if is_success_record(record) else 0,
    )


def collect_rows(folder_path, routes_file):
    file_paths = sorted(glob.glob(os.path.join(folder_path, "*.json")))
    route_catalog = load_route_catalog(routes_file)
    expected_route_ids = {row["ScenarioID"] for row in route_catalog}
    records_by_route_id = {}

    for file_path in file_paths:
        with open(file_path) as file:
            data = json.load(file)

        records = data.get("_checkpoint", {}).get("records", [])
        for record in records:
            scenario_id = extract_scenario_id(record)
            if scenario_id not in expected_route_ids:
                continue

            current_record = records_by_route_id.get(scenario_id)
            if current_record is None or record_preference_key(record) > record_preference_key(
                current_record
            ):
                records_by_route_id[scenario_id] = record

    rows = []
    missing_count = 0
    for route_info in route_catalog:
        record = records_by_route_id.get(route_info["ScenarioID"])
        if record is None:
            missing_count += 1
            rows.append(
                {
                    **route_info,
                    "Success": "X",
                    "Route Completed": 0,
                }
            )
            continue

        rows.append(
            {
                **route_info,
                "Success": "O" if is_success_record(record) else "X",
                "Route Completed": record.get("scores", {}).get("score_route", 0) or 0,
            }
        )

    return rows, missing_count


def excel_column_name(index):
    result = []
    while index > 0:
        index, remainder = divmod(index - 1, 26)
        result.append(chr(ord("A") + remainder))
    return "".join(reversed(result))


def make_inline_string_cell(cell_ref, value, style_id=0):
    text = escape("" if value is None else str(value))
    return (
        f'<c r="{cell_ref}" t="inlineStr" s="{style_id}">'
        f"<is><t>{text}</t></is></c>"
    )


def make_number_cell(cell_ref, value, style_id=0):
    return f'<c r="{cell_ref}" s="{style_id}"><v>{value}</v></c>'


def build_sheet_xml(rows):
    total_rows = len(rows) + 1
    dimension = f"A1:E{total_rows}"

    xml_rows = []

    header_cells = []
    for column_index, header in enumerate(HEADERS, start=1):
        cell_ref = f"{excel_column_name(column_index)}1"
        header_cells.append(make_inline_string_cell(cell_ref, header, style_id=1))
    xml_rows.append(f'<row r="1">{"".join(header_cells)}</row>')

    for row_index, row in enumerate(rows, start=2):
        row_cells = [
            make_number_cell(f"A{row_index}", int(row["ScenarioID"]))
            if str(row["ScenarioID"]).isdigit()
            else make_inline_string_cell(f"A{row_index}", row["ScenarioID"]),
            make_inline_string_cell(f"B{row_index}", row["Town"]),
            make_inline_string_cell(f"C{row_index}", row["Task"]),
            make_inline_string_cell(f"D{row_index}", row["Success"]),
            make_number_cell(f"E{row_index}", row["Route Completed"]),
        ]
        xml_rows.append(f'<row r="{row_index}">{"".join(row_cells)}</row>')

    return f"""<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <dimension ref="{dimension}"/>
  <sheetViews>
    <sheetView workbookViewId="0">
      <pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/>
    </sheetView>
  </sheetViews>
  <sheetFormatPr defaultRowHeight="15"/>
  <cols>
    <col min="1" max="1" width="12" customWidth="1"/>
    <col min="2" max="2" width="12" customWidth="1"/>
    <col min="3" max="3" width="34" customWidth="1"/>
    <col min="4" max="4" width="10" customWidth="1"/>
    <col min="5" max="5" width="18" customWidth="1"/>
  </cols>
  <sheetData>
    {"".join(xml_rows)}
  </sheetData>
  <autoFilter ref="{dimension}"/>
</worksheet>
"""


def build_workbook_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"
          xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">
  <sheets>
    <sheet name="Routes" sheetId="1" r:id="rId1"/>
  </sheets>
</workbook>
"""


def build_workbook_rels_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet"
                Target="worksheets/sheet1.xml"/>
  <Relationship Id="rId2"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles"
                Target="styles.xml"/>
</Relationships>
"""


def build_root_rels_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">
  <Relationship Id="rId1"
                Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument"
                Target="xl/workbook.xml"/>
</Relationships>
"""


def build_content_types_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">
  <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>
  <Default Extension="xml" ContentType="application/xml"/>
  <Override PartName="/xl/workbook.xml"
            ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>
  <Override PartName="/xl/worksheets/sheet1.xml"
            ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>
  <Override PartName="/xl/styles.xml"
            ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>
</Types>
"""


def build_styles_xml():
    return """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">
  <fonts count="2">
    <font>
      <sz val="11"/>
      <name val="Calibri"/>
      <family val="2"/>
    </font>
    <font>
      <b/>
      <sz val="11"/>
      <name val="Calibri"/>
      <family val="2"/>
    </font>
  </fonts>
  <fills count="2">
    <fill><patternFill patternType="none"/></fill>
    <fill><patternFill patternType="gray125"/></fill>
  </fills>
  <borders count="1">
    <border><left/><right/><top/><bottom/><diagonal/></border>
  </borders>
  <cellStyleXfs count="1">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0"/>
  </cellStyleXfs>
  <cellXfs count="2">
    <xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>
    <xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/>
  </cellXfs>
  <cellStyles count="1">
    <cellStyle name="Normal" xfId="0" builtinId="0"/>
  </cellStyles>
</styleSheet>
"""


def write_xlsx(output_path, rows):
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with zipfile.ZipFile(output_path, "w", compression=zipfile.ZIP_DEFLATED) as workbook:
        workbook.writestr("[Content_Types].xml", build_content_types_xml())
        workbook.writestr("_rels/.rels", build_root_rels_xml())
        workbook.writestr("xl/workbook.xml", build_workbook_xml())
        workbook.writestr("xl/_rels/workbook.xml.rels", build_workbook_rels_xml())
        workbook.writestr("xl/styles.xml", build_styles_xml())
        workbook.writestr("xl/worksheets/sheet1.xml", build_sheet_xml(rows))


def main():
    parser = argparse.ArgumentParser(
        description="Create an Excel summary from Bench2Drive route result JSON files."
    )
    parser.add_argument("-f", "--folder", required=True, help="Folder containing route result json files")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output .xlsx path (default: <folder>/hc_route_summary.xlsx)",
    )
    parser.add_argument(
        "-r",
        "--routes-file",
        default=DEFAULT_ROUTES_FILE,
        help=f"Bench2Drive routes xml path (default: {DEFAULT_ROUTES_FILE})",
    )
    args = parser.parse_args()

    output_path = args.output or os.path.join(args.folder, "hc_route_summary.xlsx")
    rows, missing_count = collect_rows(args.folder, args.routes_file)

    write_xlsx(output_path, rows)
    print(
        f"Wrote {len(rows)} rows to {output_path} "
        f"(missing routes filled with zero score: {missing_count})"
    )


if __name__ == "__main__":
    main()
