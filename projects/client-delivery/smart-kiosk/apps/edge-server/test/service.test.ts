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

  it('applies configured promotions before tax and records the final total', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database, {
      taxRateBps: 825,
      promotions: [{ code: 'WELCOME10', type: 'PERCENT_BPS', value: 1000 }]
    });

    const quote = service.quote({
      items: [{ menuItemId: 'item-fries', quantity: 2, modifiers: [] }],
      promotionCode: 'welcome10'
    });
    expect(quote).toMatchObject({ subtotalMinor: 650, discountAmountMinor: 65, taxAmountMinor: 48, totalAmountMinor: 633 });
    expect(() => service.quote({ items: [{ menuItemId: 'item-fries', quantity: 1, modifiers: [] }], promotionCode: 'EXPIRED' }))
      .toThrow('Promotion code is not available');
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

  it('preserves a declined-payment order for retry', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);
    const order = service.createOrder({
      idempotencyKey: '7a76b7bf-5fed-4a31-827c-1c8c291e42a1',
      orderType: 'TAKE_OUT',
      items: [{ menuItemId: 'item-fries', quantity: 1, modifiers: [] }]
    });

    const failed = service.startMockPayment(order.transaction_id, 'DECLINED');
    expect(failed?.order_status).toBe('AWAITING_PAYMENT');
    expect(failed?.dispatches).toHaveLength(0);

    const completed = service.startMockPayment(order.transaction_id, 'APPROVED');
    expect(completed?.order_status).toBe('COMPLETED');
    expect(completed?.payments).toHaveLength(2);
    database.close();
  });

  it('cancels an awaiting-payment order', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);
    const order = service.createOrder({
      idempotencyKey: '4c1c317b-d950-44bc-9dc3-31ea42cb4f75',
      orderType: 'TAKE_OUT',
      items: [{ menuItemId: 'item-fries', quantity: 1, modifiers: [] }]
    });

    const cancelled = service.cancelOrder(order.transaction_id);
    expect(cancelled?.order_status).toBe('CANCELLED');
    database.close();
  });

  it('lets staff take an item offline and restore it', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);

    expect(service.getMenu('schedule-lunch').items.some((item) => item.menu_item_id === 'item-fries')).toBe(true);
    expect(service.setItemAvailability('item-fries', false)).toMatchObject({ itemName: 'Seasoned Fries', isAvailable: false });
    expect(service.getMenu('schedule-lunch').items.some((item) => item.menu_item_id === 'item-fries')).toBe(false);

    expect(service.setItemAvailability('item-fries', true)).toMatchObject({ itemName: 'Seasoned Fries', isAvailable: true });
    expect(service.getMenu('schedule-lunch').items.some((item) => item.menu_item_id === 'item-fries')).toBe(true);
    database.close();
  });

  it('returns an operational staff overview', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);
    service.setItemAvailability('item-fries', false);
    const order = service.createOrder({
      idempotencyKey: '6b7a9c5c-2d50-4996-9a13-a6cdf0f0ecdc',
      orderType: 'TAKE_OUT',
      items: [{ menuItemId: 'item-cola', quantity: 1, modifiers: [] }]
    });

    const overview = service.getStaffOverview();
    expect(overview.integrationMode).toBe('mock');
    expect(overview.unavailableItems).toEqual([{ menu_item_id: 'item-fries', item_name: 'Seasoned Fries' }]);
    expect(overview.menuItems.find((item) => item.menu_item_id === 'item-fries')?.is_available).toBe(0);
    expect(overview.orderStatusCounts).toContainEqual({ status: 'AWAITING_PAYMENT', count: 1 });
    expect(overview.recentOrders[0]?.transaction_id).toBe(order.transaction_id);
    database.close();
  });

  it('lets staff update a menu price used by subsequent quotes', () => {
    const database = openDatabase(':memory:');
    const service = new KioskService(database);

    expect(service.setItemPrice('item-fries', 400)).toMatchObject({ itemName: 'Seasoned Fries', basePriceMinor: 400 });
    expect(service.quote({ items: [{ menuItemId: 'item-fries', quantity: 1, modifiers: [] }] }).totalAmountMinor).toBe(400);
    database.close();
  });
});
