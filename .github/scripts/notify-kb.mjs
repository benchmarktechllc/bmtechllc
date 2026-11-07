import { readFile } from 'node:fs/promises';
import { pathToFileURL } from 'node:url';

const branch = 'kb-notification-state';
const statePath = 'kb-delivery-state.json';

export function portalLocation(env = process.env) {
  const configured = (env.PORTAL_BASE_URL || '').trim().replace(/\/$/, '');
  if (configured) {
    const url = new URL(configured.includes('://') ? configured : `https://${configured}`);
    const pathname = url.pathname.endsWith('/') ? url.pathname : `${url.pathname}/`;
    return { origin: url.origin, prefix: pathname };
  }
  const repository = env.GITHUB_REPOSITORY || '';
  const owner = (env.GITHUB_REPOSITORY_OWNER || repository.split('/')[0] || '').trim();
  if (!/^[A-Za-z0-9-]+$/.test(owner)) {
    throw new Error('Set PORTAL_BASE_URL or GITHUB_REPOSITORY_OWNER to the current GitHub Pages owner');
  }
  return { origin: `https://${owner}.github.io`, prefix: '/bmtechllc/portal/' };
}

export function newPages(manifest, state, location = portalLocation()) {
  if (!Array.isArray(manifest) || !Array.isArray(state.announced)) {
    throw new Error('Invalid manifest or delivery history');
  }
  const seen = new Set(state.announced);
  return manifest.filter(page => {
    if (typeof page.url !== 'string' || !page.url.startsWith(location.prefix) ||
        new URL(page.url, location.origin).origin !== location.origin ||
        typeof page.title !== 'string' || typeof page.description !== 'string') {
      throw new Error('Invalid portal page');
    }
    if (page.status !== 'published' || seen.has(page.url)) return false;
    seen.add(page.url);
    return true;
  });
}

export function payload(page, location = portalLocation()) {
  return {
    username: 'BMTech KB',
    allowed_mentions: { parse: [] },
    content: '📚 New documentation published',
    embeds: [{
      title: page.title.slice(0, 256),
      url: new URL(page.url, location.origin).href,
      description: page.description.slice(0, 1500),
      color: 0x5865f2,
      footer: { text: 'BenchmarkTech Knowledge Base' },
    }],
  };
}

export async function send(webhook, page, request = fetch, sleep = ms => new Promise(r => setTimeout(r, ms)), location = portalLocation()) {
  const url = new URL(webhook);
  if (url.protocol !== 'https:' || url.hostname !== 'discord.com' ||
      !/^\/api(?:\/v\d+)?\/webhooks\/\d+\/[^/]+$/.test(url.pathname)) {
    throw new Error('DISCORD_KB_WEBHOOK_URL must be a Discord incoming webhook');
  }
  url.searchParams.set('wait', 'true');
  for (let attempt = 0; attempt < 5; attempt++) {
    let response;
    try {
      response = await request(url, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload(page, location)), signal: AbortSignal.timeout(30000),
        redirect: 'error',
      });
    } catch {
      // Do not include fetch errors: they may contain the secret webhook URL.
      throw new Error('Discord request failed; delivery may be uncertain. Inspect channel before retrying.');
    }
    if (response.status === 429) {
      const body = await response.json();
      const seconds = Number(body.retry_after);
      if (!Number.isFinite(seconds) || seconds < 0 || seconds > 120) throw new Error('Invalid Discord retry delay');
      await sleep(Math.ceil(seconds * 1000) + 250);
      continue;
    }
    if (!response.ok) throw new Error(`Discord returned HTTP ${response.status}`);
    const message = await response.json();
    if (!message.id) throw new Error('Discord did not confirm a message ID');
    return message.id;
  }
  throw new Error('Discord rate limit persisted; retry the failed job later');
}

export async function main(manifestPath) {
  const { GH_TOKEN: token, GITHUB_REPOSITORY: repository, DISCORD_KB_WEBHOOK_URL: webhook } = process.env;
  if (!token || !repository || !webhook) throw new Error('Configure DISCORD_KB_WEBHOOK_URL and GitHub Actions credentials');
  const api = async (path, options = {}) => {
    const response = await fetch(`https://api.github.com/repos/${repository}/${path}`, {
      ...options,
      headers: { Authorization: `Bearer ${token}`, Accept: 'application/vnd.github+json',
        'Content-Type': 'application/json', 'X-GitHub-Api-Version': '2022-11-28' },
      signal: AbortSignal.timeout(30000),
    });
    if (response.status === 404) return null;
    if (!response.ok) throw new Error(`GitHub state request failed: HTTP ${response.status}`);
    return response.json();
  };
  const manifest = JSON.parse(await readFile(manifestPath, 'utf8'));
  let stored = await api(`contents/${statePath}?ref=${branch}`);
  let state = stored ? JSON.parse(Buffer.from(stored.content, 'base64').toString())
    : JSON.parse(await readFile(new URL('./kb-baseline.json', import.meta.url), 'utf8'));
  const location = portalLocation();
  const pages = newPages(manifest, state, location);
  const save = async () => {
    const result = await api(`contents/${statePath}`, {
      method: 'PUT', body: JSON.stringify({
        branch, message: 'Record KB notification deliveries',
        content: Buffer.from(JSON.stringify(state, null, 2) + '\n').toString('base64'),
        ...(stored ? { sha: stored.sha } : {}),
      }),
    });
    if (!result?.content?.sha) throw new Error('Delivery history was not saved');
    stored = result.content;
  };
  if (!stored) {
    if (!await api(`git/ref/heads/${branch}`)) {
      const ref = await api('git/ref/heads/main');
      if (!ref) throw new Error('Cannot find main branch');
      await api('git/refs', { method: 'POST', body: JSON.stringify({ ref: `refs/heads/${branch}`, sha: ref.object.sha }) });
    }
    // Verify write access before posting anything.
    await save();
  }
  for (const page of pages) {
    const messageId = await send(webhook, page, fetch, ms => new Promise(r => setTimeout(r, ms)), location);
    state.announced.push(page.url);
    state.lastDelivery = { url: page.url, messageId, at: new Date().toISOString() };
    await save();
    console.log(`Announced ${page.url}`);
  }
  console.log(`Delivered ${pages.length} new KB page notification(s).`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main(process.argv[2]).catch(error => {
    console.error(error.message);
    process.exitCode = 1;
  });
}
