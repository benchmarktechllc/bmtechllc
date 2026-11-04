import { describe, expect, it } from 'vitest';
import { openDatabase } from '../src/database.js';
import { KioskService } from '../src/service.js';

describe('KioskService', () => {
  it('offers burger and drink items in the local menu', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);

    const menu = service.getMenu('schedule-lunch');

    expect(menu.categories.map((category) => category.category_name)).toEqual(['Burgers', 'Sides', 'Drinks']);
    expect(menu.items.filter((item) => item.category_id === 'category-burgers')).toHaveLength(3);
    expect(menu.items.filter((item) => item.category_id === 'category-drinks')).toHaveLength(3);
    database.close();
  });

  it('prices the Truffle Bistro Burger double size option', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);

    const quote = service.quote({
      items: [{ menuItemId: 'item-truffle-burger', quantity: 1, modifiers: [{ modifierId: 'modifier-truffle-double', quantity: 1 }] }]
    });

    expect(quote.totalAmountMinor).toBe(2050);
    database.close();
  });

  it('prices the double size option on each burger item', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);

    const quote = service.quote({
      items: [{ menuItemId: 'item-burger', quantity: 1, modifiers: [{ modifierId: 'modifier-truffle-double', quantity: 1 }] }]
    });

    expect(quote.totalAmountMinor).toBe(1225);
    database.close();
  });

  it('creates an idempotent order and completes mock dispatches after approval', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);
    const request = {
      idempotencyKey: '0d8d1621-fd2e-47f6-b6e8-8cc4a2aa5e4d',
      orderType: 'TAKE_OUT' as const,
      items: [{ menuItemId: 'item-burger', quantity: 1, modifiers: [{ modifierId: 'modifier-cheese', quantity: 1 }] }]
    };

    const first = service.createOrder(request);
    const repeated = service.createOrder(request);
    expect(repeated.transaction_id).toBe(first.transaction_id);
    expect(first.total_amount_minor).toBe(925);

    const completed = service.startMockPayment(first.transaction_id, 'APPROVED');
    expect(completed?.order_status).toBe('COMPLETED');
    expect(completed?.dispatches).toHaveLength(2);
    expect(completed?.dispatches.every((dispatch) => dispatch.dispatch_status === 'ACKNOWLEDGED')).toBe(true);
    database.close();
  });

  it('preserves a failed-payment order for retry', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);
    const order = service.createOrder({
      idempotencyKey: '7a76b7bf-5fed-4a31-827c-1c8c291e42a1',
      orderType: 'TAKE_OUT',
      items: [{ menuItemId: 'item-fries', quantity: 1, modifiers: [] }]
    });

    const failed = service.startMockPayment(order.transaction_id, 'DECLINED');
    expect(failed?.order_status).toBe('PAYMENT_FAILED');
    expect(failed?.dispatches).toHaveLength(0);
    database.close();
  });
});
