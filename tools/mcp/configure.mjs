// Install only the two optional project MCP entries; preserve unrelated client settings.
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const directory = path.join(root, '.codex');
const target = path.join(directory, 'config.toml');
const start = '# BEGIN SupportPilot optional engineering MCP';
const end = '# END SupportPilot optional engineering MCP';
await mkdir(directory, { recursive: true });
let previous = '';
try { previous = await readFile(target, 'utf8'); } catch (error) {
  if (error.code !== 'ENOENT') throw error;
}
const begin = previous.indexOf(start);
if (begin >= 0) {
  const finish = previous.indexOf(end, begin);
  if (finish < 0) throw new Error('Incomplete managed MCP block; preserve and inspect manually');
  previous = previous.slice(0, begin) + previous.slice(finish + end.length);
} else if (/\[mcp_servers\.(shrimp|structurizr)\]/.test(previous)) {
  throw new Error('Existing MCP entries require manual reconciliation; no configuration overwritten');
}
const block = `${start}
[mcp_servers.shrimp]
command = "node"
args = [${JSON.stringify(path.join(root, 'tools/mcp/shrimp.mjs'))}]
cwd = ${JSON.stringify(root)}
enabled = false
startup_timeout_sec = 30
tool_timeout_sec = 60
enabled_tools = ["plan_task", "analyze_task", "reflect_task", "split_tasks", "list_tasks", "execute_task", "verify_task", "update_task", "query_task", "get_task_detail"]

[mcp_servers.structurizr]
url = "http://127.0.0.1:3010/mcp"
enabled = false
startup_timeout_sec = 30
tool_timeout_sec = 60
enabled_tools = ["validate", "parse", "inspect"]
${end}
`;
await writeFile(target, previous.trimEnd() + '\n\n' + block);
console.log('Configured two disabled-by-default project MCP entries; existing settings preserved.');
