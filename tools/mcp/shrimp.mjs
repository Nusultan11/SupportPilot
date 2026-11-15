// Local upstream adapter: stable storage across client roots, single writer, limited tools.
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { Server } from '@modelcontextprotocol/sdk/server/index.js';
import { StdioServerTransport } from '@modelcontextprotocol/sdk/server/stdio.js';
import {
  CallToolRequestSchema, ListToolsRequestSchema, ListRootsRequestSchema,
} from '@modelcontextprotocol/sdk/types.js';
import { mkdir, open, unlink } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const data = path.join(root, 'artifacts', 'engineering', 'shrimp');
await mkdir(data, { recursive: true });
const lock = path.join(data, 'writer.lock');
let handle;
try {
  handle = await open(lock, 'wx');
} catch (error) {
  console.error(`Shrimp writer lock refused: ${lock}. Check the owning PID before recovery.`);
  throw error;
}
await handle.writeFile(JSON.stringify({ pid: process.pid, started: new Date().toISOString() }));
const allowed = {
  plan_task: 'Plan the authorized scope; AGENTS.md remains the only instruction source.',
  analyze_task: 'Analyze implementation scope and evidence.',
  reflect_task: 'Review a proposed task plan against acceptance criteria.',
  split_tasks: 'Create/update internal tasks. Use append or selective; never clear history.',
  list_tasks: 'List persistent internal tasks and their states.',
  execute_task: 'Start an eligible internal task after dependencies complete.',
  verify_task: 'Record verification evidence. Score >=80 completes an internal task only.',
  update_task: 'Update an unfinished internal task and its continuation notes.',
  query_task: 'Find tasks by query or ID.',
  get_task_detail: 'Read a task, dependencies, verification criteria and results.',
};
const upstream = new Client({ name: 'supportpilot-shrimp-adapter', version: '1.0.0' }, {
  capabilities: { roots: { listChanged: false } },
});
upstream.setRequestHandler(ListRootsRequestSchema, async () => ({ roots: [] }));
const server = new Server({ name: 'supportpilot-shrimp', version: '1.0.0' }, {
  capabilities: { tools: {} },
});
let upstreamTransport;
let requests = Promise.resolve();
let cleaning;
function cleanup() {
  return cleaning ??= (async () => {
    await requests.catch(() => {});
    if (upstreamTransport?.pid) {
      try { process.kill(upstreamTransport.pid, 'SIGTERM'); } catch { /* Already exited. */ }
    }
    await upstream.close();
    await handle.close();
    await unlink(lock);
  })();
}
try {
  // Return no roots: upstream otherwise appends a client-root basename to DATA_DIR.
  upstreamTransport = new StdioClientTransport({
    command: process.execPath,
    args: [path.join(root, 'tools/mcp/node_modules/mcp-shrimp-task-manager/dist/index.js')],
    cwd: path.join(root, 'tools', 'mcp'),
    env: {
      PATH: process.env.PATH, SystemRoot: process.env.SystemRoot,
      TEMP: process.env.TEMP, TMP: process.env.TMP,
      DATA_DIR: data, TEMPLATES_USE: 'en', ENABLE_GUI: 'false',
    },
    stderr: 'inherit',
  });
  await upstream.connect(upstreamTransport);
  server.setRequestHandler(ListToolsRequestSchema, async () => {
    const result = await upstream.listTools();
    return { tools: result.tools.filter(tool => Object.hasOwn(allowed, tool.name)).map(tool => ({
      ...tool, description: allowed[tool.name],
    })) };
  });
  server.setRequestHandler(CallToolRequestSchema, async ({ params }) => {
    if (!Object.hasOwn(allowed, params.name)) throw new Error('Tool outside project allowlist');
    // Bound trusted local requests before the upstream Zod 3 constrained-collection parser.
    if (Buffer.byteLength(JSON.stringify(params)) > 262144) {
      throw new Error('Task request exceeds local size limits');
    }
    if (params.name === 'split_tasks') {
      const tasks = JSON.parse(params.arguments?.tasksRaw ?? 'null');
      if (!Array.isArray(tasks) || tasks.length > 100) {
        throw new Error('Expected at most 100 internal tasks per request');
      }
    }
    if (params.name === 'split_tasks' &&
      !['append', 'selective'].includes(params.arguments?.updateMode)) {
      throw new Error('Only append/selective task updates preserve project history');
    }
    const result = requests.then(async () => {
      const response = await upstream.callTool(params);
      if (response.content?.some(item => item.type === 'text' &&
        item.text.startsWith('Error occurred:'))) return { ...response, isError: true };
      return response;
    });
    requests = result.then(() => {}, () => {});
    return await result;
  });
  server.onclose = () => { void cleanup(); };
  process.stdin.on('end', () => { void cleanup().then(() => process.exit(0)); });
  for (const signal of ['SIGINT', 'SIGTERM']) {
    process.on(signal, () => { void cleanup().then(() => process.exit(0)); });
  }
  await server.connect(new StdioServerTransport());
} catch (error) {
  await cleanup();
  throw error;
}
