import { existsSync } from 'node:fs';
import { spawn } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../', import.meta.url));
const python = path.join(root, 'backend', '.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
const commands = {
  start: ['-m', 'uvicorn', 'backend.app:app', '--host', '127.0.0.1', '--port', '8787'],
  ingest: ['-m', 'backend.build_menswear'],
  index: ['-m', 'backend.build_menswear'],
  test: ['-m', 'unittest', 'discover', '-s', 'backend/tests', '-v'],
  evaluate: ['-m', 'backend.menswear_validation'],
  experiment: ['-m', 'backend.experiments'],
  'research-audit': ['-m', 'backend.menswear_research_audit', 'audit'],
  check: ['-m', 'backend.tests.check_api'],
  'photo-setup': ['-m', 'backend.setup_foreground'],
};
const command = commands[process.argv[2]];
if (!command || !existsSync(python)) {
  console.error(!command ? 'Unknown backend command.' : 'backend/.venv가 필요합니다. README의 서버 설치 절차를 확인해주세요.');
  process.exit(1);
}
const child = spawn(python, ['-X', 'utf8', ...command], { cwd: root, stdio: 'inherit', shell: false });
child.on('error', (error) => { console.error(error.message); process.exitCode = 1; });
child.on('exit', (code) => { process.exitCode = code ?? 1; });
