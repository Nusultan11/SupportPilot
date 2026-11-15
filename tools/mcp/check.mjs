// Live protocol checks. Product data, model inference and task statuses are never modified.
import assert from 'node:assert/strict';
import { execFile } from 'node:child_process';
import { promisify } from 'node:util';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { ListRootsRequestSchema } from '@modelcontextprotocol/sdk/types.js';

const root = fileURLToPath(new URL('../../', import.meta.url));
const reportRoot = path.join(root, 'artifacts/reports/engineering/mcp');
const attempt = new Date().toISOString().replaceAll(':', '-');
const evidence = path.join(reportRoot, 'attempts', attempt);
await mkdir(evidence, { recursive: true });
const sha = bytes => createHash('sha256').update(bytes).digest('hex');
async function save(name, result) {
  await writeFile(path.join(evidence, name + '.json'), JSON.stringify(result, null, 2) + '\n');
}
function text(result) {
  assert.notEqual(result.isError, true);
  return result.content.filter(item => item.type === 'text').map(item => item.text).join('\n');
}
async function connect(server) {
  const client = new Client({ name: 'supportpilot-live-verification', version: '1.0.0' }, {
    capabilities: { roots: { listChanged: false } },
  });
  client.setRequestHandler(ListRootsRequestSchema, async () => ({
    roots: [{ uri: new URL('file:///different-client-root').href, name: 'different-client-root' }],
  }));
  const transport = server === 'shrimp'
    ? new StdioClientTransport({
      command: process.execPath, args: [path.join(root, 'tools/mcp/shrimp.mjs')],
      cwd: root, stderr: 'pipe',
    })
    : new StreamableHTTPClientTransport(new URL('http://127.0.0.1:3010/mcp'));
  await client.connect(transport);
  return client;
}
const report = { checked_at: new Date().toISOString(), checks: {}, limitations: [] };
let client;
try {
  client = await connect('shrimp');
  const tools = await client.listTools();
  await save('shrimp-tools', tools);
  assert.equal(tools.tools.length, 10);
  assert(!tools.tools.some(tool => ['delete_task', 'clear_all_tasks', 'init_project_rules'].includes(tool.name)));
  const before = await readFile(path.join(root, 'artifacts/engineering/shrimp/tasks.json'));
  text(await client.callTool({ name: 'list_tasks', arguments: { status: 'all' } }));
  for (const [name, args] of [
    ['clear_all_tasks', {}],
    ['split_tasks', { updateMode: 'clearAllTasks', tasksRaw: '[]' }],
    ['split_tasks', { updateMode: 'overwrite', tasksRaw: '[]' }],
    ['plan_task', { description: 'x'.repeat(262145) }],
  ]) {
    const result = await client.callTool({ name, arguments: args }).catch(error => ({ isError: true, error: error.message }));
    assert.equal(result.isError, true);
  }
  const competing = await promisify(execFile)(process.execPath, [
    'tools/mcp/client.mjs', 'shrimp', 'list',
  ], { cwd: root, timeout: 10000, windowsHide: true }).then(() => null, error => error);
  assert(competing?.stderr.includes('writer lock refused'));
  assert.equal(sha(await readFile(path.join(root, 'artifacts/engineering/shrimp/tasks.json'))), sha(before));
  await client.close();
  client = await connect('shrimp');
  text(await client.callTool({ name: 'list_tasks', arguments: { status: 'all' } }));
  assert.equal(sha(await readFile(path.join(root, 'artifacts/engineering/shrimp/tasks.json'))), sha(before));
  report.checks.shrimp = {
    tools: 10, cross_process_persistence: true, client_roots_do_not_change_store: true,
    destructive_calls_rejected: true, oversized_call_rejected: true,
    competing_writer_rejected: true, task_state_sha256: sha(before),
  };
  await client.close();
  client = await connect('structurizr');
  const tools2 = await client.listTools();
  await save('structurizr-tools', tools2);
  assert.deepEqual(tools2.tools.map(tool => tool.name).sort(), ['inspect', 'parse', 'validate']);
  const dsl = await readFile(path.join(root, 'docs/architecture/workspace.dsl'), 'utf8');
  let workspace;
  for (const operation of ['validate', 'parse', 'inspect']) {
    const result = await client.callTool({ name: operation, arguments: { dsl } });
    await save(operation, result);
    const value = text(result);
    if (operation === 'validate') assert.equal(value, 'OK');
    if (operation === 'parse') workspace = JSON.parse(value);
    if (operation === 'inspect') {
      const findings = JSON.parse(value);
      assert(findings.every(finding => finding.startsWith('INFO:')));
      report.checks.inspection_findings = findings;
    }
  }
  assert.equal(workspace.views.systemContextViews.length, 1);
  assert.equal(workspace.views.containerViews.length, 1);
  assert.equal(workspace.views.componentViews.length, 2);
  const components = workspace.model.softwareSystems.flatMap(system => system.containers ?? [])
    .flatMap(container => container.components ?? []);
  assert(components.some(component => component.tags.includes('Planned')));
  const invalid = await client.callTool({ name: 'validate', arguments: { dsl: 'workspace { invalid' } });
  await save('invalid-dsl', invalid);
  assert.notEqual(invalid.content?.[0]?.text, 'OK');
  report.checks.structurizr = {
    validate: 'OK', parse: true, invalid_dsl_rejected: true, components: components.length,
    context_views: 1, container_views: 1, component_views: 2, dsl_sha256: sha(dsl),
  };
  report.limitations = [
    'Upstream Zod 3 advisory remains; trusted local stdio only, request bytes/task count bounded.',
    'Upstream task writes are not transactional; single writer and verified backups are required.',
    'Two INFO findings: owning architecture/ADR Markdown stays external to the DSL payload.',
    'Live protocol/client checks do not hot-reload tools into an already-running host chat.',
  ];
  report.status = 'PASSED';
} catch (error) {
  report.status = 'FAILED';
  report.error = error.stack;
  throw error;
} finally {
  await client?.close();
  await save('live-check', report);
  await writeFile(path.join(reportRoot, 'latest-check.json'), JSON.stringify({
    status: report.status, receipt: path.relative(root, path.join(evidence, 'live-check.json')),
  }, null, 2) + '\n');
}
console.log(JSON.stringify(report, null, 2));
