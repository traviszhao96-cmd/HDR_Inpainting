"""Replay saved cases without changing their images, masks, or parameters."""
import json
import time
import uuid
from pathlib import Path

import requests


def main():
    root = Path(__file__).resolve().parents[1]
    run = root / 'test_runs' / (time.strftime('%Y%m%d-%H%M%S') + '-' + uuid.uuid4().hex[:6])
    run.mkdir(parents=True)
    results = []
    print(f'RUN {run}', flush=True)
    for case_file in sorted((root / 'test_masks').glob('*/case.json')):
        case = json.loads(case_file.read_text())
        print(f'START {case_file.parent.name} {case["source_name"]}', flush=True)
        entry = {'case': str(case_file), 'source_name': case['source_name']}
        try:
            with (case_file.parent / case['image']).open('rb') as image, (case_file.parent / case['mask']).open('rb') as mask:
                response = requests.post('http://127.0.0.1:7860/erase',
                    files={'image': (case['source_name'], image), 'mask': ('mask.png', mask, 'image/png')},
                    data={key: case[key] for key in ('model', 'prompt', 'mask_expand')}, timeout=360)
            response.raise_for_status()
            entry['result'] = response.json()
            print(f'OK {entry["result"]["job_id"]} {entry["result"]["elapsed_ms"]}ms', flush=True)
        except (requests.RequestException, ValueError) as error:
            entry['error'] = str(error)
            print(f'ERROR {error}', flush=True)
        results.append(entry)
        (run / 'results.json').write_text(json.dumps(results, ensure_ascii=False, indent=2))
    print(f'COMPLETE {len(results)} cases: {run / "results.json"}', flush=True)


if __name__ == '__main__':
    main()
