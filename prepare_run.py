"""Create an isolated copy for a research run. Never execute or delete data."""
from pathlib import Path
import argparse, datetime, json, shutil, hashlib
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--run-id',default=datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
a=p.parse_args()
if not a.run_id or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.run_id):p.error('Use letters, digits, _ and - only')
root=Path(__file__).resolve().parent
out=root/'runs'/a.run_id
if out.exists():p.error('Run already exists; choose a new id')
out.mkdir(parents=True)
for name in ['src','scripts','research','requirements.txt']:
 s=root/name
 if s.is_dir():shutil.copytree(s,out/name)
 elif s.is_file():shutil.copy2(s,out/name)
for n in ['work','outputs']:(out/n).mkdir(exist_ok=True)
manifest=[]
for f in out.rglob('*'):
 if f.is_file():manifest.append({'file':str(f.relative_to(out)),'sha256':hashlib.sha256(f.read_bytes()).hexdigest()})
(out/'run_manifest.json').write_text(json.dumps({'run_id':a.run_id,'executed':False,'files':manifest},indent=2),encoding='utf-8')
print(out)
print('No research was executed. Prepare inputs and follow README. No automatic cleanup.')
