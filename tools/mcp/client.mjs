// Reproducible MCP client for environments that cannot reload the agent's tool catalogue.
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { StdioClientTransport } from '@modelcontextprotocol/sdk/client/stdio.js';
import { StreamableHTTPClientTransport } from '@modelcontextprotocol/sdk/client/streamableHttp.js';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import path from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const [server, operation = 'list', input, output] = process.argv.slice(2);
if (!['shrimp', 'structurizr'].includes(server)) {
  throw new Error('Usage: node tools/mcp/client.mjs shrimp|structurizr [list|tool input.json output.json]');
}
const client = new Client({ name: 'supportpilot-engineering-check', version: '1.0.0' });
const transport = server === 'shrimp'
  ? new StdioClientTransport({
    command: process.execPath, args: [path.join(root, 'tools/mcp/shrimp.mjs')],
    cwd: root, stderr: 'inherit',
  })
  : new StreamableHTTPClientTransport(new URL('http://127.0.0.1:3010/mcp'));
try {
  await client.connect(transport);
  const result = operation === 'list'
    ? await client.listTools()
    : await client.callTool({
      name: operation, arguments: JSON.parse(await readFile(input, 'utf8')),
    });
  if (output) {
    await mkdir(path.dirname(output), { recursive: true });
    await writeFile(output, JSON.stringify(result, null, 2) + '\n');
    console.log(JSON.stringify({ server, operation, isError: result.isError ?? false, output }));
  } else {
    console.log(JSON.stringify(result, null, 2));
  }
  if (result.isError) process.exitCode = 1;
} finally {
  await client.close();
}
