import { randomUUID, timingSafeEqual } from 'node:crypto';
import { resolve } from 'node:path';
import cors from '@fastify/cors';
import Fastify from 'fastify';
import { z } from 'zod';
import {
  createOrderRequestSchema,
  quoteRequestSchema,
  staffItemAvailabilityRequestSchema,
  staffMenuItemRequestSchema,
  staffItemPriceRequestSchema,
  startPaymentRequestSchema
} from '@smart-kiosk/shared';
import { openDatabase } from './database.js';
import { KioskService, type Promotion } from './service.js';

const app = Fastify({ logger: true });
const databasePath = process.env.EDGE_DATABASE_PATH ?? resolve(process.cwd(), 'data/smart-kiosk.sqlite');
const service = new KioskService(openDatabase(databasePath), {
  taxRateBps: readBoundedInteger(process.env.TAX_RATE_BPS, 0, 10_000),
  promotions: readPromotions(process.env.KIOSK_PROMOTIONS_JSON)
});
const staffAccessToken = process.env.STAFF_ACCESS_TOKEN
  ?? (process.env.NODE_ENV === 'production' ? undefined : 'dev-staff-token');
const kioskEventSchema = z.object({
  eventType: z.enum(['order_started', 'item_added', 'upsell_viewed', 'upsell_added', 'checkout_viewed', 'payment_started']),
  context: z.record(z.string(), z.unknown()).optional()
});

await app.register(cors, { origin: process.env.KIOSK_ORIGIN ?? 'http://localhost:5173' });

app.addHook('onRequest', async (request, reply) => {
  reply.header('X-Request-Id', request.id);
  reply.header('X-Content-Type-Options', 'nosniff');
  reply.header('X-Frame-Options', 'DENY');
  reply.header('Referrer-Policy', 'no-referrer');
  reply.header('Cache-Control', 'no-store');
});

app.get('/', async () => ({
  service: 'Smart Kiosk Edge API',
  integrationMode: 'mock',
  kioskUrl: process.env.KIOSK_ORIGIN ?? 'http://localhost:5173',
  healthUrl: '/health',
  menuUrl: '/v1/menu'
}));
app.get('/health', async () => ({ status: 'ok', integrationMode: 'mock' }));
app.get('/ready', async (_request, reply) => {
  try {
    service.isReady();
    return { status: 'ready' };
  } catch {
    return reply.code(503).send({ status: 'not_ready' });
  }
});
app.get('/v1/menu', async (request) => {
  const query = request.query as { scheduleId?: string };
  return service.getMenu(query.scheduleId);
});

app.post('/v1/telemetry/events', async (request, reply) => {
  const parsed = kioskEventSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: 'INVALID_EVENT', details: parsed.error.flatten() });
  return reply.code(202).send(service.recordKioskEvent(parsed.data.eventType, parsed.data.context));
});

app.post('/v1/quotes', async (request, reply) => {
  const parsed = quoteRequestSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: 'INVALID_QUOTE', details: parsed.error.flatten() });
  try {
    return service.quote(parsed.data);
  } catch (error) {
    return reply.code(422).send({ error: 'UNPRICEABLE_CART', message: error instanceof Error ? error.message : 'Unable to price cart' });
  }
});

app.post('/v1/orders', async (request, reply) => {
  const parsed = createOrderRequestSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: 'INVALID_ORDER', details: parsed.error.flatten() });
  try {
    return reply.code(201).send(service.createOrder(parsed.data));
  } catch (error) {
    return reply.code(422).send({ error: 'ORDER_REJECTED', message: error instanceof Error ? error.message : 'Unable to create order' });
  }
});

app.get('/v1/orders/:transactionId', async (request, reply) => {
  const { transactionId } = request.params as { transactionId: string };
  const order = service.getOrder(transactionId);
  return order ? order : reply.code(404).send({ error: 'ORDER_NOT_FOUND' });
});

app.post('/v1/orders/:transactionId/payment', async (request, reply) => {
  const parsed = startPaymentRequestSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: 'INVALID_PAYMENT', details: parsed.error.flatten() });
  const { transactionId } = request.params as { transactionId: string };
  try {
    return service.startMockPayment(transactionId, parsed.data.scenario);
  } catch (error) {
    return reply.code(409).send({ error: 'PAYMENT_REJECTED', message: error instanceof Error ? error.message : 'Unable to start payment' });
  }
});

app.post('/v1/orders/:transactionId/cancel', async (request, reply) => {
  const { transactionId } = request.params as { transactionId: string };
  try {
    return service.cancelOrder(transactionId);
  } catch (error) {
    return reply.code(409).send({ error: 'CANCEL_REJECTED', message: error instanceof Error ? error.message : 'Unable to cancel order' });
  }
});

app.get('/v1/staff/overview', async (request, reply) => {
  if (!isStaffAuthorized(request.headers['x-staff-token'])) return reply.code(401).send({ error: 'STAFF_AUTH_REQUIRED' });
  return service.getStaffOverview();
});

app.post('/v1/staff/menu-items/:menuItemId/availability', async (request, reply) => {
  if (!isStaffAuthorized(request.headers['x-staff-token'])) return reply.code(401).send({ error: 'STAFF_AUTH_REQUIRED' });
  const parsed = staffItemAvailabilityRequestSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: 'INVALID_AVAILABILITY', details: parsed.error.flatten() });
  const { menuItemId } = request.params as { menuItemId: string };
  try {
    return service.setItemAvailability(menuItemId, parsed.data.isAvailable);
  } catch (error) {
    return reply.code(404).send({ error: 'MENU_ITEM_NOT_FOUND', message: error instanceof Error ? error.message : 'Menu item not found' });
  }
});

app.post('/v1/staff/menu-items/:menuItemId/price', async (request, reply) => {
  if (!isStaffAuthorized(request.headers['x-staff-token'])) return reply.code(401).send({ error: 'STAFF_AUTH_REQUIRED' });
  const parsed = staffItemPriceRequestSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: 'INVALID_PRICE', details: parsed.error.flatten() });
  const { menuItemId } = request.params as { menuItemId: string };
  try {
    return service.setItemPrice(menuItemId, parsed.data.basePriceMinor);
  } catch (error) {
    return reply.code(404).send({ error: 'MENU_ITEM_NOT_FOUND', message: error instanceof Error ? error.message : 'Menu item not found' });
  }
});

app.post('/v1/staff/menu-items', async (request, reply) => {
  if (!isStaffAuthorized(request.headers['x-staff-token'])) return reply.code(401).send({ error: 'STAFF_AUTH_REQUIRED' });
  const parsed = staffMenuItemRequestSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: 'INVALID_MENU_ITEM', details: parsed.error.flatten() });
  try {
    return reply.code(201).send(service.createMenuItem(parsed.data));
  } catch (error) {
    return reply.code(409).send({ error: 'MENU_ITEM_CREATE_REJECTED', message: error instanceof Error ? error.message : 'Unable to add menu item' });
  }
});

app.put('/v1/staff/menu-items/:menuItemId', async (request, reply) => {
  if (!isStaffAuthorized(request.headers['x-staff-token'])) return reply.code(401).send({ error: 'STAFF_AUTH_REQUIRED' });
  const parsed = staffMenuItemRequestSchema.safeParse(request.body);
  if (!parsed.success) return reply.code(400).send({ error: 'INVALID_MENU_ITEM', details: parsed.error.flatten() });
  const { menuItemId } = request.params as { menuItemId: string };
  try {
    return service.updateMenuItem(menuItemId, parsed.data);
  } catch (error) {
    return reply.code(409).send({ error: 'MENU_ITEM_UPDATE_REJECTED', message: error instanceof Error ? error.message : 'Unable to update menu item' });
  }
});

app.post('/v1/staff/orders/:transactionId/cancel', async (request, reply) => {
  if (!isStaffAuthorized(request.headers['x-staff-token'])) return reply.code(401).send({ error: 'STAFF_AUTH_REQUIRED' });
  const { transactionId } = request.params as { transactionId: string };
  try {
    return service.cancelOrder(transactionId, 'staff-console');
  } catch (error) {
    return reply.code(409).send({ error: 'CANCEL_REJECTED', message: error instanceof Error ? error.message : 'Unable to cancel order' });
  }
});

app.setErrorHandler((error, _request, reply) => {
  app.log.error({ err: error, requestId: randomUUID() }, 'Unhandled edge-service error');
  reply.code(500).send({ error: 'INTERNAL_ERROR' });
});

const host = process.env.EDGE_HOST ?? '127.0.0.1';
const port = Number(process.env.EDGE_PORT ?? 3001);
await app.listen({ host, port });

function isStaffAuthorized(value: string | string[] | undefined): boolean {
  if (!staffAccessToken || typeof value !== 'string') return false;
  const expected = Buffer.from(staffAccessToken);
  const received = Buffer.from(value);
  return expected.length === received.length && timingSafeEqual(expected, received);
}

function readBoundedInteger(value: string | undefined, fallback: number, maximum: number): number {
  if (value === undefined) return fallback;
  const parsed = Number(value);
  if (!Number.isInteger(parsed) || parsed < 0 || parsed > maximum) throw new Error(`Configuration value must be an integer from 0 to ${maximum}`);
  return parsed;
}

function readPromotions(value: string | undefined): Promotion[] {
  if (!value) return [];
  const parsed: unknown = JSON.parse(value);
  if (!Array.isArray(parsed)) throw new Error('KIOSK_PROMOTIONS_JSON must be a JSON array');
  return parsed.map((promotion) => {
    if (!promotion || typeof promotion !== 'object') throw new Error('Each promotion must be an object');
    const candidate = promotion as Record<string, unknown>;
    const code = typeof candidate.code === 'string' ? candidate.code.trim().toUpperCase() : '';
    const type = candidate.type;
    const amount = candidate.value;
    if (!code || !['PERCENT_BPS', 'AMOUNT_MINOR'].includes(String(type)) || !Number.isInteger(amount) || Number(amount) <= 0) {
      throw new Error('Promotions require a code, supported type, and positive integer value');
    }
    return { code, type: type as Promotion['type'], value: Number(amount) };
  });
}
