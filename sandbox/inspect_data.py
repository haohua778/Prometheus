import csv
import json
import sys
from pathlib import Path


def inspect(path: Path, file_format: str) -> dict:
    with path.open(encoding='utf-8', newline='') as source:
        if file_format == 'csv':
            reader = csv.DictReader(source)
            fields = reader.fieldnames
            if not fields or any(not field for field in fields) or len(set(fields)) != len(fields):
                raise ValueError('CSV must have unique, nonempty headers.')
            samples = []
            count = 0
            for count, row in enumerate(reader, start=1):
                if None in row or any(value is None for value in row.values()):
                    raise ValueError('CSV row does not match the headers.')
                if len(samples) < 5:
                    samples.append(row)
        else:
            records = json.load(source, parse_constant=reject_constant)
            if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
                raise ValueError('JSON must be an array of record objects.')
            fields = list(dict.fromkeys(key for row in records for key in row))
            count = len(records)
            samples = records[:5]
    return {'fields': fields, 'record_count': count, 'samples': samples}


def reject_constant(value: str):
    raise ValueError('Non-finite JSON numbers are not supported.')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', newline='\n')
    try:
        result = inspect(Path(sys.argv[1]), sys.argv[2])
    except (ValueError, UnicodeError, csv.Error, OSError, RecursionError) as error:
        print(json.dumps({'error': str(error)}))
        sys.exit(2)
    output = json.dumps(result, ensure_ascii=False, allow_nan=False)
    if len(output.encode('utf-8')) + 1 > 16384:
        print(json.dumps({'error': 'Inspection output exceeds 16 KiB.'}))
        sys.exit(3)
    print(output)
