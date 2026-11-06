import { z } from 'zod';

export const moneySchema = z.number().int().nonnegative();

export const modifierSelectionSchema = z.object({
  modifierId: z.string().min(1),
  quantity: z.number().int().positive().default(1)
});

export const cartLineSchema = z.object({
  menuItemId: z.string().min(1),
  quantity: z.number().int().positive().max(20),
  modifiers: z.array(modifierSelectionSchema).default([])
});

export const quoteRequestSchema = z.object({
  items: z.array(cartLineSchema).min(1),
  promotionCode: z.string().trim().min(1).max(32).optional()
});

export const paymentScenarioSchema = z.enum([
  'APPROVED',
  'DECLINED',
  'CANCELLED',
  'TIMED_OUT',
  'FAILED'
]);

export const orderStatusSchema = z.enum([
  'DRAFT',
  'AWAITING_PAYMENT',
  'PAID',
  'FULFILLMENT_PENDING',
  'COMPLETED',
  'CANCELLED',
  'PAYMENT_FAILED'
]);

export const paymentStatusSchema = z.enum([
  'INITIATED',
  'PENDING',
  'APPROVED',
  'DECLINED',
  'CANCELLED',
  'FAILED',
  'TIMED_OUT',
  'UNKNOWN'
]);

export const createOrderRequestSchema = quoteRequestSchema.extend({
  idempotencyKey: z.string().uuid(),
  orderType: z.enum(['DINE_IN', 'TAKE_OUT']).default('TAKE_OUT')
});

export const startPaymentRequestSchema = z.object({
  scenario: paymentScenarioSchema
});

export const staffItemAvailabilityRequestSchema = z.object({
  isAvailable: z.boolean()
});

export const staffItemPriceRequestSchema = z.object({
  basePriceMinor: moneySchema
});

export type CartLine = z.infer<typeof cartLineSchema>;
export type QuoteRequest = z.infer<typeof quoteRequestSchema>;
export type CreateOrderRequest = z.infer<typeof createOrderRequestSchema>;
export type PaymentScenario = z.infer<typeof paymentScenarioSchema>;
export type PaymentStatus = z.infer<typeof paymentStatusSchema>;
export type StaffItemAvailabilityRequest = z.infer<typeof staffItemAvailabilityRequestSchema>;
export type StaffItemPriceRequest = z.infer<typeof staffItemPriceRequestSchema>;

export interface PricedModifier {
  modifierId: string;
  groupId: string;
  name: string;
  sku: string;
  priceDeltaMinor: number;
  quantity: number;
}

export interface PricedLine {
  menuItemId: string;
  name: string;
  sku: string;
  quantity: number;
  baseUnitPriceMinor: number;
  lineTotalMinor: number;
  modifiers: PricedModifier[];
}

export interface OrderQuote {
  currencyCode: string;
  subtotalMinor: number;
  discountAmountMinor: number;
  taxAmountMinor: number;
  totalAmountMinor: number;
  lines: PricedLine[];
}
