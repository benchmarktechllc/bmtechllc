import { randomUUID } from 'node:crypto';
import { resolve } from 'node:path';
import cors from '@fastify/cors';
import Fastify from 'fastify';
import {
  createOrderRequestSchema,
  quoteRequestSchema,
  startPaymentRequestSchema
} from '@smart-kiosk/shared';
import { openDatabase } from './database.js';
import { KioskService } from './service.js';

const app = Fastify({ logger: true });
const databasePath = process.env.EDGE_DATABASE_PATH ?? resolve(process.cwd(), 'data/smart-kiosk.sqlite');
const service = new KioskService(openDatabase(databasePath));

await app.register(cors, { origin: process.env.KIOSK_ORIGIN ?? 'http://localhost:5173' });

app.get('/', async () => ({
  service: 'Smart Kiosk Edge API',
  integrationMode: 'mock',
  kioskUrl: process.env.KIOSK_ORIGIN ?? 'http://localhost:5173',
  healthUrl: '/health',
  menuUrl: '/v1/menu'
}));
app.get('/health', async () => ({ status: 'ok', integrationMode: 'mock' }));
app.get('/v1/menu', async (request) => {
  const query = request.query as { scheduleId?: string };
  return service.getMenu(query.scheduleId);
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

app.setErrorHandler((error, _request, reply) => {
  app.log.error({ err: error, requestId: randomUUID() }, 'Unhandled edge-service error');
  reply.code(500).send({ error: 'INTERNAL_ERROR' });
});

const host = process.env.EDGE_HOST ?? '127.0.0.1';
const port = Number(process.env.EDGE_PORT ?? 3001);
await app.listen({ host, port });
