import test from 'node:test';
import assert from 'node:assert/strict';
import { newPages, payload, portalLocation, send } from './notify-kb.mjs';
const page = { url: '/bmtechllc/portal/reference/new/', title: 'New article', description: 'A summary', status: 'published' };
const webhook = 'https://discord.com/api/webhooks/123/test-token';
const location = { origin: 'https://example-org.github.io', prefix: '/bmtechllc/portal/' };

test('announces new pages and draft promotions; ignores edits, drafts, archives and duplicates', () => {
  const old = { ...page, url: '/bmtechllc/portal/reference/old/' };
  assert.deepEqual(newPages([old, page, page, { ...page, url: `${page.url}draft/`, status: 'draft' },
    { ...page, url: `${page.url}archive/`, status: 'archived' }], { announced: [old.url] }, location), [page]);
  assert.deepEqual(newPages([page], { announced: [page.url] }, location), []);
  assert.deepEqual(newPages([page], { announced: [] }, location), [page]);
});
test('rejects malformed manifest and foreign URLs', () => {
  assert.throws(() => newPages({}, { announced: [] }, location));
  assert.throws(() => newPages([{ ...page, url: 'https://example.com/' }], { announced: [] }, location));
});
test('portal host follows the current repository owner', () => {
  assert.deepEqual(portalLocation({ GITHUB_REPOSITORY_OWNER: 'benchmarktechllc' }), {
    origin: 'https://benchmarktechllc.github.io',
    prefix: '/bmtechllc/portal/',
  });
  assert.equal(
    portalLocation({ PORTAL_BASE_URL: 'https://pages.example/bmtechllc/portal' }).origin,
    'https://pages.example',
  );
  assert.throws(() => portalLocation({}));
});
test('payload includes live link, bounded text and no mentions', () => {
  const result = payload({ ...page, title: 'a'.repeat(400), description: '@everyone'.repeat(1000) }, location);
  assert.equal(result.embeds[0].url, `https://example-org.github.io${page.url}`);
  assert.equal(result.embeds[0].title.length, 256);
  assert.equal(result.embeds[0].description.length, 1500);
  assert.deepEqual(result.allowed_mentions, { parse: [] });
});
test('waits for confirmation and respects rate limits', async () => {
  let calls = 0;
  const pauses = [];
  const id = await send(webhook, page, async (url, options) => {
    assert.equal(url.searchParams.get('wait'), 'true');
    assert.equal(options.method, 'POST');
    return ++calls === 1 ? new Response(JSON.stringify({ retry_after: 0.5 }), { status: 429 })
      : new Response(JSON.stringify({ id: 'message-123' }));
  }, async ms => pauses.push(ms), location);
  assert.equal(id, 'message-123');
  assert.deepEqual(pauses, [750]);
});
test('failures do not expose webhook secret or retry ambiguous deliveries', async () => {
  let calls = 0;
  await assert.rejects(send(webhook, page, async () => { calls++; throw new Error(webhook); }, undefined, location), error => !error.message.includes('test-token'));
  assert.equal(calls, 1);
  await assert.rejects(send(webhook, page, async () => new Response('{}', { status: 401 }), undefined, location), /HTTP 401/);
  await assert.rejects(send(webhook, page, async () => new Response('{}'), undefined, location), /message ID/);
});

test('persists confirmed deliveries and suppresses them on rerun', async () => {
  const { main } = await import('./notify-kb.mjs');
  const { mkdtemp, writeFile, rm } = await import('node:fs/promises');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  const directory = await mkdtemp(join(tmpdir(), 'kb-test-'));
  const file = join(directory, 'index.json');
  await writeFile(file, JSON.stringify([page]));
  const savedFetch = globalThis.fetch;
  const keys = ['GH_TOKEN', 'GITHUB_REPOSITORY', 'GITHUB_REPOSITORY_OWNER', 'PORTAL_BASE_URL', 'DISCORD_KB_WEBHOOK_URL'];
  const savedEnv = keys.map(key => process.env[key]);
  Object.assign(process.env, { GH_TOKEN: 'test', GITHUB_REPOSITORY: 'test/repo', DISCORD_KB_WEBHOOK_URL: webhook });
  delete process.env.PORTAL_BASE_URL;
  delete process.env.GITHUB_REPOSITORY_OWNER;
  let state = null;
  let messages = 0;
  let writes = 0;
  globalThis.fetch = async (url, options = {}) => {
    const address = String(url);
    if (address.startsWith('https://discord.com')) {
      assert.ok(state, 'state must be writable before sending');
      messages++;
      return new Response(JSON.stringify({ id: 'confirmed' }));
    }
    if (options.method === 'PUT') {
      const body = JSON.parse(options.body);
      state = { content: body.content, sha: `state-${++writes}` };
      return new Response(JSON.stringify({ content: { sha: state.sha } }));
    }
    if (address.includes('/contents/')) return new Response(JSON.stringify(state), { status: state ? 200 : 404 });
    if (address.includes('/git/ref/heads/kb-')) return new Response('{}', { status: 404 });
    if (address.endsWith('/git/ref/heads/main')) return new Response(JSON.stringify({ object: { sha: 'main-sha' } }));
    if (address.endsWith('/git/refs')) return new Response('{}');
    throw new Error('Unexpected request');
  };
  try {
    await main(file);
    await main(file);
    assert.equal(messages, 1);
    assert.equal(writes, 2);
    const persisted = JSON.parse(Buffer.from(state.content, 'base64'));
    assert.equal(persisted.lastDelivery.messageId, 'confirmed');
    assert.ok(persisted.announced.includes(page.url));
  } finally {
    globalThis.fetch = savedFetch;
    keys.forEach((key, i) => savedEnv[i] === undefined ? delete process.env[key] : process.env[key] = savedEnv[i]);
    await rm(directory, { recursive: true });
  }
});
