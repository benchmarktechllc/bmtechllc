import { randomUUID } from 'node:crypto';
import type { DatabaseSync } from 'node:sqlite';
import type {
  CartLine,
  CreateOrderRequest,
  OrderQuote,
  PaymentScenario,
  PaymentStatus,
  PricedLine,
  PricedModifier,
  QuoteRequest
} from '@smart-kiosk/shared';

const storeId = process.env.STORE_ID ?? 'store-demo-001';
const terminalId = process.env.TERMINAL_ID ?? 'kiosk-demo-001';
const registerId = 'register-kiosk-01';

type MenuRow = {
  menu_item_id: string;
  item_name: string;
  sku: string;
  base_price_minor: number;
};

type ModifierRow = {
  modifier_id: string;
  modifier_group_id: string;
  modifier_name: string;
  modifier_sku: string;
  price_adjustment_minor: number;
};

type StoredOrder = Record<string, unknown> & {
  transaction_id: string;
  order_status: string;
  total_amount_minor: number;
};

export type Promotion = {
  code: string;
  type: 'PERCENT_BPS' | 'AMOUNT_MINOR';
  value: number;
};

type PricingOptions = {
  taxRateBps?: number;
  promotions?: Promotion[];
};

export class KioskService {
  private readonly taxRateBps: number;
  private readonly promotions: Map<string, Promotion>;

  constructor(private readonly db: DatabaseSync, options: PricingOptions = {}) {
    this.taxRateBps = options.taxRateBps ?? 0;
    this.promotions = new Map((options.promotions ?? []).map((promotion) => [promotion.code.trim().toUpperCase(), promotion]));
  }

  getMenu(scheduleId?: string) {
    const schedules = this.db
      .prepare('SELECT schedule_id, schedule_name, start_time, end_time FROM menu_schedules WHERE store_id = ? AND is_active = 1')
      .all(storeId) as Array<Record<string, unknown>>;
    const selectedSchedule = scheduleId ?? this.currentScheduleId(schedules);
    const categories = this.db
      .prepare(
        `SELECT DISTINCT c.category_id, c.category_name, c.display_order
         FROM menu_categories c
         JOIN menu_items i ON i.category_id = c.category_id
         JOIN menu_item_schedule ms ON ms.menu_item_id = i.menu_item_id
         WHERE c.is_active = 1 AND i.is_active = 1 AND ms.schedule_id = ?
         ORDER BY c.display_order`
      )
      .all(selectedSchedule) as Array<Record<string, unknown>>;
    const items = this.db
      .prepare(
        `SELECT i.menu_item_id, i.category_id, i.item_name, i.sku, i.base_price_minor
         FROM menu_items i
         JOIN menu_item_schedule ms ON ms.menu_item_id = i.menu_item_id
         WHERE i.is_active = 1 AND ms.schedule_id = ?
         AND NOT EXISTS (
           SELECT 1 FROM store_item_availability_overrides o
           WHERE o.store_id = ? AND o.menu_item_id = i.menu_item_id
             AND o.cleared_at IS NULL AND o.is_available = 0
         )
         ORDER BY i.item_name`
      )
      .all(selectedSchedule, storeId) as Array<Record<string, unknown>>;

    return { storeId, currencyCode: 'USD', activeScheduleId: selectedSchedule, schedules, categories, items };
  }

  quote(request: QuoteRequest): OrderQuote {
    const lines = request.items.map((line) => this.priceLine(line));
    const subtotalMinor = lines.reduce((sum, line) => sum + line.lineTotalMinor, 0);
    const promotion = request.promotionCode ? this.promotions.get(request.promotionCode.trim().toUpperCase()) : undefined;
    if (request.promotionCode && !promotion) throw new Error('Promotion code is not available');
    const discountAmountMinor = promotion
      ? Math.min(subtotalMinor, promotion.type === 'PERCENT_BPS'
        ? Math.round(subtotalMinor * promotion.value / 10_000)
        : promotion.value)
      : 0;
    const taxableAmountMinor = subtotalMinor - discountAmountMinor;
    const taxAmountMinor = Math.round(taxableAmountMinor * this.taxRateBps / 10_000);
    return { currencyCode: 'USD', subtotalMinor, discountAmountMinor, taxAmountMinor, totalAmountMinor: taxableAmountMinor + taxAmountMinor, lines };
  }

  createOrder(request: CreateOrderRequest) {
    const existing = this.db
      .prepare('SELECT transaction_id FROM orders WHERE idempotency_key = ?')
      .get(request.idempotencyKey) as { transaction_id: string } | undefined;
    if (existing) {
      const priorOrder = this.getOrder(existing.transaction_id);
      if (priorOrder) return priorOrder;
      throw new Error('Idempotency record has no order');
    }

    const quote = this.quote(request);
    const transactionId = randomUUID();
    const now = new Date().toISOString();
    this.db.exec('BEGIN IMMEDIATE');
    try {
      this.db
        .prepare(
          `INSERT INTO orders (
            transaction_id, idempotency_key, terminal_id, store_id, virtual_register_id,
            order_type, order_status, currency_code, subtotal_minor, discount_amount_minor, tax_amount_minor,
            total_amount_minor, created_at, updated_at
          ) VALUES (?, ?, ?, ?, ?, ?, 'AWAITING_PAYMENT', ?, ?, ?, ?, ?, ?, ?)`
        )
        .run(
          transactionId,
          request.idempotencyKey,
          terminalId,
          storeId,
          registerId,
          request.orderType,
          quote.currencyCode,
          quote.subtotalMinor,
          quote.discountAmountMinor,
          quote.taxAmountMinor,
          quote.totalAmountMinor,
          now,
          now
        );
      for (const line of quote.lines) this.insertLine(transactionId, line);
      this.db.exec('COMMIT');
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }
    const createdOrder = this.getOrder(transactionId);
    if (!createdOrder) throw new Error('Created order could not be read');
    return createdOrder;
  }

  startMockPayment(transactionId: string, scenario: PaymentScenario) {
    const order = this.getOrder(transactionId);
    if (!order) throw new Error('Order not found');
    if (order.order_status !== 'AWAITING_PAYMENT') throw new Error('Order is not awaiting payment');

    const statusByScenario: Record<PaymentScenario, PaymentStatus> = {
      APPROVED: 'APPROVED',
      DECLINED: 'DECLINED',
      CANCELLED: 'CANCELLED',
      TIMED_OUT: 'TIMED_OUT',
      FAILED: 'FAILED'
    };
    const paymentStatus = statusByScenario[scenario];
    const now = new Date().toISOString();
    const nextAttempt =
      ((this.db.prepare('SELECT COUNT(*) as count FROM payments WHERE transaction_id = ?').get(transactionId) as { count: number })
        .count ?? 0) + 1;
    this.db.exec('BEGIN IMMEDIATE');
    try {
      this.db
        .prepare(
          'INSERT INTO payments (payment_id, transaction_id, attempt_number, payment_status, amount_minor, is_simulated, created_at, authorized_at) VALUES (?, ?, ?, ?, ?, 1, ?, ?)'
        )
        .run(randomUUID(), transactionId, nextAttempt, paymentStatus, order.total_amount_minor, now, paymentStatus === 'APPROVED' ? now : null);
      if (paymentStatus === 'APPROVED') {
        this.db.prepare("UPDATE orders SET order_status = 'FULFILLMENT_PENDING', updated_at = ? WHERE transaction_id = ?").run(now, transactionId);
        this.createDispatch(transactionId, 'KITCHEN', 'mock-kitchen', order, now);
        this.createDispatch(transactionId, 'POS', 'mock-pos', order, now);
        this.db.prepare("UPDATE integration_dispatches SET dispatch_status = 'ACKNOWLEDGED', sent_at = ?, last_response_at = ? WHERE transaction_id = ? AND destination_type = 'KITCHEN'").run(now, now, transactionId);
        this.db.prepare("UPDATE integration_dispatches SET dispatch_status = 'ACKNOWLEDGED', sent_at = ?, last_response_at = ? WHERE transaction_id = ? AND destination_type = 'POS'").run(now, now, transactionId);
        this.db.prepare("UPDATE orders SET order_status = 'COMPLETED', updated_at = ? WHERE transaction_id = ?").run(now, transactionId);
      } else {
        // A terminal response is a failed attempt, not a failed order. Keeping the
        // order available for payment lets the kiosk offer a clear retry path.
        this.db.prepare("UPDATE orders SET order_status = 'AWAITING_PAYMENT', updated_at = ? WHERE transaction_id = ?").run(now, transactionId);
      }
      this.db.exec('COMMIT');
    } catch (error) {
      this.db.exec('ROLLBACK');
      throw error;
    }
    return this.getOrder(transactionId);
  }

  cancelOrder(transactionId: string) {
    const order = this.getOrder(transactionId);
    if (!order) throw new Error('Order not found');
    if (order.order_status !== 'AWAITING_PAYMENT') throw new Error('Order can no longer be cancelled');

    this.db.prepare("UPDATE orders SET order_status = 'CANCELLED', updated_at = ? WHERE transaction_id = ?")
      .run(new Date().toISOString(), transactionId);
    return this.getOrder(transactionId);
  }

  getOrder(transactionId: string) {
    const order = this.db
      .prepare('SELECT * FROM orders WHERE transaction_id = ?')
      .get(transactionId) as StoredOrder | undefined;
    if (!order) return undefined;
    const items = this.db.prepare('SELECT * FROM order_items WHERE transaction_id = ?').all(transactionId) as Array<Record<string, unknown>>;
    const payments = this.db.prepare('SELECT * FROM payments WHERE transaction_id = ? ORDER BY attempt_number').all(transactionId) as Array<Record<string, unknown>>;
    const dispatches = this.db.prepare('SELECT * FROM integration_dispatches WHERE transaction_id = ?').all(transactionId) as Array<Record<string, unknown>>;
    return { ...order, items, payments, dispatches };
  }

  getStaffOverview() {
    const orderStatusCounts = this.db.prepare(
      'SELECT order_status as status, COUNT(*) as count FROM orders GROUP BY order_status ORDER BY order_status'
    ).all() as Array<{ status: string; count: number }>;
    const dispatchStatusCounts = this.db.prepare(
      'SELECT dispatch_status as status, COUNT(*) as count FROM integration_dispatches GROUP BY dispatch_status ORDER BY dispatch_status'
    ).all() as Array<{ status: string; count: number }>;
    const unavailableItems = this.db.prepare(
      `SELECT i.menu_item_id, i.item_name
       FROM store_item_availability_overrides o
       JOIN menu_items i ON i.menu_item_id = o.menu_item_id
       WHERE o.store_id = ? AND o.cleared_at IS NULL AND o.is_available = 0
       ORDER BY i.item_name`
    ).all(storeId) as Array<{ menu_item_id: string; item_name: string }>;
    const menuItems = this.db.prepare(
      `SELECT i.menu_item_id, i.item_name, i.base_price_minor,
       CASE WHEN EXISTS (
         SELECT 1 FROM store_item_availability_overrides o
         WHERE o.store_id = ? AND o.menu_item_id = i.menu_item_id
           AND o.cleared_at IS NULL AND o.is_available = 0
       ) THEN 0 ELSE 1 END AS is_available
       FROM menu_items i WHERE i.is_active = 1 ORDER BY i.item_name`
    ).all(storeId) as Array<{ menu_item_id: string; item_name: string; base_price_minor: number; is_available: number }>;
    const recentOrders = this.db.prepare(
      `SELECT transaction_id, order_status, total_amount_minor, created_at
       FROM orders ORDER BY created_at DESC LIMIT 10`
    ).all() as Array<{ transaction_id: string; order_status: string; total_amount_minor: number; created_at: string }>;

    return {
      generatedAt: new Date().toISOString(),
      integrationMode: 'mock' as const,
      orderStatusCounts,
      dispatchStatusCounts,
      unavailableItems,
      menuItems,
      recentOrders
    };
  }

  setItemAvailability(menuItemId: string, isAvailable: boolean) {
    const item = this.db.prepare('SELECT menu_item_id, item_name FROM menu_items WHERE menu_item_id = ?').get(menuItemId) as MenuRow | undefined;
    if (!item) throw new Error('Menu item not found');

    const now = new Date().toISOString();
    const activeOverride = this.db.prepare(
      'SELECT override_id FROM store_item_availability_overrides WHERE store_id = ? AND menu_item_id = ? AND cleared_at IS NULL'
    ).get(storeId, menuItemId) as { override_id: string } | undefined;

    if (isAvailable) {
      this.db.prepare(
        'UPDATE store_item_availability_overrides SET cleared_at = ? WHERE store_id = ? AND menu_item_id = ? AND cleared_at IS NULL'
      ).run(now, storeId, menuItemId);
    } else if (activeOverride) {
      this.db.prepare('UPDATE store_item_availability_overrides SET is_available = 0 WHERE override_id = ?').run(activeOverride.override_id);
    } else {
      this.db.prepare(
        'INSERT INTO store_item_availability_overrides (override_id, store_id, menu_item_id, is_available, cleared_at) VALUES (?, ?, ?, 0, NULL)'
      ).run(randomUUID(), storeId, menuItemId);
    }

    return { menuItemId: item.menu_item_id, itemName: item.item_name, isAvailable };
  }

  setItemPrice(menuItemId: string, basePriceMinor: number) {
    const item = this.db.prepare('SELECT menu_item_id, item_name FROM menu_items WHERE menu_item_id = ?').get(menuItemId) as MenuRow | undefined;
    if (!item) throw new Error('Menu item not found');
    this.db.prepare('UPDATE menu_items SET base_price_minor = ? WHERE menu_item_id = ?').run(basePriceMinor, menuItemId);
    return { menuItemId: item.menu_item_id, itemName: item.item_name, basePriceMinor };
  }

  isReady(): boolean {
    this.db.prepare('SELECT 1 as ready').get();
    return true;
  }

  private priceLine(line: CartLine): PricedLine {
    const item = this.db
      .prepare('SELECT menu_item_id, item_name, sku, base_price_minor FROM menu_items WHERE menu_item_id = ? AND is_active = 1')
      .get(line.menuItemId) as MenuRow | undefined;
    if (!item) throw new Error(`Unavailable menu item: ${line.menuItemId}`);
    const modifiers = line.modifiers.map((selection) => {
      const modifier = this.db
        .prepare(
          `SELECT m.modifier_id, m.modifier_group_id, m.modifier_name, m.modifier_sku, m.price_adjustment_minor
           FROM modifiers m
           JOIN menu_item_modifier_groups mg ON mg.modifier_group_id = m.modifier_group_id
           WHERE m.modifier_id = ? AND mg.menu_item_id = ? AND m.is_active = 1`
        )
        .get(selection.modifierId, line.menuItemId) as ModifierRow | undefined;
      if (!modifier) throw new Error(`Invalid modifier: ${selection.modifierId}`);
      const priced: PricedModifier = {
        modifierId: modifier.modifier_id,
        groupId: modifier.modifier_group_id,
        name: modifier.modifier_name,
        sku: modifier.modifier_sku,
        priceDeltaMinor: modifier.price_adjustment_minor,
        quantity: selection.quantity
      };
      return priced;
    });
    const modifierTotal = modifiers.reduce((sum, modifier) => sum + modifier.priceDeltaMinor * modifier.quantity, 0);
    return {
      menuItemId: item.menu_item_id,
      name: item.item_name,
      sku: item.sku,
      quantity: line.quantity,
      baseUnitPriceMinor: item.base_price_minor,
      lineTotalMinor: (item.base_price_minor + modifierTotal) * line.quantity,
      modifiers
    };
  }

  private insertLine(transactionId: string, line: PricedLine): void {
    const orderItemId = randomUUID();
    this.db
      .prepare(
        'INSERT INTO order_items (order_item_id, transaction_id, menu_item_id, quantity, unit_price_minor, line_total_minor, item_name_snapshot, item_sku_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
      )
      .run(orderItemId, transactionId, line.menuItemId, line.quantity, line.baseUnitPriceMinor, line.lineTotalMinor, line.name, line.sku);
    for (const modifier of line.modifiers) {
      this.db
        .prepare(
          'INSERT INTO order_item_modifiers (order_item_modifier_id, order_item_id, modifier_id, modifier_group_id_snapshot, quantity, modifier_price_minor, modifier_name_snapshot, modifier_sku_snapshot) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
        )
        .run(randomUUID(), orderItemId, modifier.modifierId, modifier.groupId, modifier.quantity, modifier.priceDeltaMinor, modifier.name, modifier.sku);
    }
  }

  private createDispatch(transactionId: string, destinationType: 'KITCHEN' | 'POS', destinationId: string, order: Record<string, unknown>, now: string): void {
    this.db
      .prepare(
        'INSERT INTO integration_dispatches (dispatch_id, transaction_id, destination_type, destination_id, idempotency_key, payload_json, dispatch_status, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)'
      )
      .run(randomUUID(), transactionId, destinationType, destinationId, `${transactionId}:${destinationType}`, JSON.stringify({ transactionId, order, simulated: true }), 'PENDING', now);
  }

  private currentScheduleId(schedules: Array<Record<string, unknown>>): string {
    const hourMinute = new Intl.DateTimeFormat('en-GB', {
      timeZone: 'America/Chicago',
      hour: '2-digit',
      minute: '2-digit',
      hourCycle: 'h23'
    }).format(new Date());
    const selected = schedules.find((schedule) => {
      const start = String(schedule.start_time);
      const end = String(schedule.end_time);
      return hourMinute >= start && hourMinute <= end;
    });
    return String(selected?.schedule_id ?? schedules[0]?.schedule_id ?? 'schedule-lunch');
  }
}
