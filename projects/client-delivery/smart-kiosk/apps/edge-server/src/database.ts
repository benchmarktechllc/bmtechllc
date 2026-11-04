import { DatabaseSync } from 'node:sqlite';
import { mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

const schema = `
  PRAGMA foreign_keys = ON;
  CREATE TABLE IF NOT EXISTS stores (
    store_id TEXT PRIMARY KEY,
    store_name TEXT NOT NULL,
    timezone TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS kiosk_terminals (
    terminal_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    terminal_name TEXT NOT NULL,
    created_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS pos_registers (
    virtual_register_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    register_code TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
  );
  CREATE TABLE IF NOT EXISTS menu_categories (
    category_id TEXT PRIMARY KEY,
    category_name TEXT NOT NULL,
    display_order INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
  );
  CREATE TABLE IF NOT EXISTS menu_items (
    menu_item_id TEXT PRIMARY KEY,
    category_id TEXT NOT NULL REFERENCES menu_categories(category_id),
    item_name TEXT NOT NULL,
    sku TEXT NOT NULL UNIQUE,
    base_price_minor INTEGER NOT NULL CHECK (base_price_minor >= 0),
    is_active INTEGER NOT NULL DEFAULT 1
  );
  CREATE TABLE IF NOT EXISTS modifier_groups (
    modifier_group_id TEXT PRIMARY KEY,
    group_name TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS modifiers (
    modifier_id TEXT PRIMARY KEY,
    modifier_group_id TEXT NOT NULL REFERENCES modifier_groups(modifier_group_id),
    modifier_name TEXT NOT NULL,
    modifier_sku TEXT NOT NULL UNIQUE,
    price_adjustment_minor INTEGER NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
  );
  CREATE TABLE IF NOT EXISTS menu_item_modifier_groups (
    menu_item_id TEXT NOT NULL REFERENCES menu_items(menu_item_id),
    modifier_group_id TEXT NOT NULL REFERENCES modifier_groups(modifier_group_id),
    display_order INTEGER NOT NULL,
    min_selections INTEGER NOT NULL DEFAULT 0 CHECK (min_selections >= 0),
    max_selections INTEGER NOT NULL CHECK (max_selections >= min_selections),
    PRIMARY KEY (menu_item_id, modifier_group_id)
  );
  CREATE TABLE IF NOT EXISTS menu_schedules (
    schedule_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    schedule_name TEXT NOT NULL,
    start_time TEXT NOT NULL,
    end_time TEXT NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1
  );
  CREATE TABLE IF NOT EXISTS menu_item_schedule (
    schedule_id TEXT NOT NULL REFERENCES menu_schedules(schedule_id),
    menu_item_id TEXT NOT NULL REFERENCES menu_items(menu_item_id),
    PRIMARY KEY (schedule_id, menu_item_id)
  );
  CREATE TABLE IF NOT EXISTS store_item_availability_overrides (
    override_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    menu_item_id TEXT NOT NULL REFERENCES menu_items(menu_item_id),
    is_available INTEGER NOT NULL,
    cleared_at TEXT
  );
  CREATE UNIQUE INDEX IF NOT EXISTS active_item_override
    ON store_item_availability_overrides(store_id, menu_item_id)
    WHERE cleared_at IS NULL;
  CREATE TABLE IF NOT EXISTS store_modifier_availability_overrides (
    override_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    modifier_id TEXT NOT NULL REFERENCES modifiers(modifier_id),
    is_available INTEGER NOT NULL,
    cleared_at TEXT
  );
  CREATE UNIQUE INDEX IF NOT EXISTS active_modifier_override
    ON store_modifier_availability_overrides(store_id, modifier_id)
    WHERE cleared_at IS NULL;
  CREATE TABLE IF NOT EXISTS orders (
    transaction_id TEXT PRIMARY KEY,
    idempotency_key TEXT NOT NULL UNIQUE,
    terminal_id TEXT NOT NULL REFERENCES kiosk_terminals(terminal_id),
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    virtual_register_id TEXT NOT NULL REFERENCES pos_registers(virtual_register_id),
    order_type TEXT NOT NULL CHECK (order_type IN ('DINE_IN', 'TAKE_OUT')),
    order_status TEXT NOT NULL,
    currency_code TEXT NOT NULL,
    subtotal_minor INTEGER NOT NULL,
    tax_amount_minor INTEGER NOT NULL,
    tip_amount_minor INTEGER NOT NULL DEFAULT 0,
    total_amount_minor INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS order_items (
    order_item_id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES orders(transaction_id),
    menu_item_id TEXT NOT NULL REFERENCES menu_items(menu_item_id),
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_minor INTEGER NOT NULL,
    line_total_minor INTEGER NOT NULL,
    item_name_snapshot TEXT NOT NULL,
    item_sku_snapshot TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS order_item_modifiers (
    order_item_modifier_id TEXT PRIMARY KEY,
    order_item_id TEXT NOT NULL REFERENCES order_items(order_item_id),
    modifier_id TEXT NOT NULL REFERENCES modifiers(modifier_id),
    modifier_group_id_snapshot TEXT NOT NULL,
    quantity INTEGER NOT NULL CHECK (quantity > 0),
    modifier_price_minor INTEGER NOT NULL,
    modifier_name_snapshot TEXT NOT NULL,
    modifier_sku_snapshot TEXT NOT NULL
  );
  CREATE TABLE IF NOT EXISTS payments (
    payment_id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES orders(transaction_id),
    attempt_number INTEGER NOT NULL,
    payment_status TEXT NOT NULL,
    amount_minor INTEGER NOT NULL,
    is_simulated INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    authorized_at TEXT,
    UNIQUE(transaction_id, attempt_number)
  );
  CREATE TABLE IF NOT EXISTS integration_dispatches (
    dispatch_id TEXT PRIMARY KEY,
    transaction_id TEXT NOT NULL REFERENCES orders(transaction_id),
    destination_type TEXT NOT NULL CHECK (destination_type IN ('KITCHEN', 'POS')),
    destination_id TEXT NOT NULL,
    idempotency_key TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    dispatch_status TEXT NOT NULL,
    retry_count INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT,
    last_error TEXT,
    created_at TEXT NOT NULL,
    sent_at TEXT,
    last_response_at TEXT,
    UNIQUE(transaction_id, destination_type, destination_id)
  );
  CREATE TABLE IF NOT EXISTS offline_payment_queue (
    queue_id TEXT PRIMARY KEY,
    payment_id TEXT NOT NULL REFERENCES payments(payment_id),
    transaction_uuid TEXT NOT NULL,
    payload TEXT NOT NULL,
    queue_status TEXT NOT NULL,
    queued_at TEXT NOT NULL
  );
`;

export function openDatabase(path: string): DatabaseSync {
  if (path !== ':memory:') mkdirSync(dirname(path), { recursive: true });
  const db = new DatabaseSync(path);
  db.exec('PRAGMA foreign_keys = ON; CREATE TABLE IF NOT EXISTS schema_migrations (migration_id TEXT PRIMARY KEY, applied_at TEXT NOT NULL);');
  const applied = db.prepare('SELECT migration_id FROM schema_migrations WHERE migration_id = ?').get('001_initial_schema');
  if (!applied) {
    db.exec('BEGIN IMMEDIATE');
    try {
      db.exec(schema);
      db.prepare('INSERT INTO schema_migrations (migration_id, applied_at) VALUES (?, ?)').run('001_initial_schema', new Date().toISOString());
      db.exec('COMMIT');
    } catch (error) {
      db.exec('ROLLBACK');
      throw error;
    }
  }
  seedDatabase(db);
  return db;
}

function seedDatabase(db: DatabaseSync): void {
  const now = new Date().toISOString();
  db.prepare(
    'INSERT OR IGNORE INTO stores (store_id, store_name, timezone, created_at) VALUES (?, ?, ?, ?)'
  ).run('store-demo-001', 'Demo Store', 'America/Chicago', now);
  db.prepare(
    'INSERT OR IGNORE INTO kiosk_terminals (terminal_id, store_id, terminal_name, created_at) VALUES (?, ?, ?, ?)'
  ).run('kiosk-demo-001', 'store-demo-001', 'Demo Kiosk', now);
  db.prepare(
    'INSERT OR IGNORE INTO pos_registers (virtual_register_id, store_id, register_code) VALUES (?, ?, ?)'
  ).run('register-kiosk-01', 'store-demo-001', 'REGISTER_KIOSK_01');

  const categories = [
    ['category-breakfast', 'Breakfast', 1],
    ['category-burgers', 'Burgers', 2],
    ['category-sides', 'Sides', 3],
    ['category-drinks', 'Drinks', 4]
  ];
  for (const category of categories) {
    db.prepare(
      'INSERT OR IGNORE INTO menu_categories (category_id, category_name, display_order) VALUES (?, ?, ?)'
    ).run(...category);
  }

  db.prepare('UPDATE menu_items SET category_id = ? WHERE menu_item_id = ?').run('category-burgers', 'item-burger');

  const items = [
    ['item-breakfast-wrap', 'category-breakfast', 'Breakfast Wrap', 'BRK-WRAP', 650],
    ['item-burger', 'category-burgers', 'Classic Burger', 'LUNCH-BURGER', 825],
    ['item-truffle-burger', 'category-burgers', 'Truffle Bistro Burger', 'BURGER-TRUFFLE', 1650],
    ['item-double-smash-burger', 'category-burgers', 'Double Smash Burger', 'BURGER-DOUBLE', 1295],
    ['item-fries', 'category-sides', 'Seasoned Fries', 'SIDE-FRIES', 325],
    ['item-cola', 'category-drinks', 'Cola', 'DRINK-COLA', 250],
    ['item-lemonade', 'category-drinks', 'Fresh Lemonade', 'DRINK-LEMONADE', 325],
    ['item-iced-tea', 'category-drinks', 'Iced Tea', 'DRINK-TEA', 275]
  ];
  for (const item of items) {
    db.prepare(
      'INSERT OR IGNORE INTO menu_items (menu_item_id, category_id, item_name, sku, base_price_minor) VALUES (?, ?, ?, ?, ?)'
    ).run(...item);
  }

  db.prepare('INSERT OR IGNORE INTO modifier_groups (modifier_group_id, group_name) VALUES (?, ?)').run(
    'group-cheese',
    'Add cheese'
  );
  db.prepare(
    'INSERT OR IGNORE INTO modifiers (modifier_id, modifier_group_id, modifier_name, modifier_sku, price_adjustment_minor) VALUES (?, ?, ?, ?, ?)'
  ).run('modifier-cheese', 'group-cheese', 'Cheddar', 'MOD-CHEDDAR', 100);
  db.prepare(
    'INSERT OR IGNORE INTO menu_item_modifier_groups (menu_item_id, modifier_group_id, display_order, min_selections, max_selections) VALUES (?, ?, ?, ?, ?)'
  ).run('item-burger', 'group-cheese', 1, 0, 1);

  db.prepare('INSERT OR IGNORE INTO modifier_groups (modifier_group_id, group_name) VALUES (?, ?)').run(
    'group-truffle-size',
    'Size'
  );
  db.prepare(
    'INSERT OR IGNORE INTO modifiers (modifier_id, modifier_group_id, modifier_name, modifier_sku, price_adjustment_minor) VALUES (?, ?, ?, ?, ?)'
  ).run('modifier-truffle-double', 'group-truffle-size', 'Double', 'MOD-TRUFFLE-DOUBLE', 400);
  db.prepare(
    'INSERT OR IGNORE INTO menu_item_modifier_groups (menu_item_id, modifier_group_id, display_order, min_selections, max_selections) VALUES (?, ?, ?, ?, ?)'
  ).run('item-truffle-burger', 'group-truffle-size', 1, 0, 1);
  for (const menuItemId of ['item-burger', 'item-double-smash-burger']) {
    db.prepare(
      'INSERT OR IGNORE INTO menu_item_modifier_groups (menu_item_id, modifier_group_id, display_order, min_selections, max_selections) VALUES (?, ?, ?, ?, ?)'
    ).run(menuItemId, 'group-truffle-size', 1, 0, 1);
  }

  const schedules = [
    ['schedule-breakfast', 'Breakfast', '00:00', '10:59'],
    ['schedule-lunch', 'Lunch', '11:00', '23:59']
  ];
  for (const schedule of schedules) {
    db.prepare(
      'INSERT OR IGNORE INTO menu_schedules (schedule_id, store_id, schedule_name, start_time, end_time) VALUES (?, ?, ?, ?, ?)'
    ).run(schedule[0]!, 'store-demo-001', schedule[1]!, schedule[2]!, schedule[3]!);
  }
  for (const mapping of [
    ['schedule-breakfast', 'item-breakfast-wrap'],
    ['schedule-breakfast', 'item-fries'],
    ['schedule-breakfast', 'item-burger'],
    ['schedule-breakfast', 'item-truffle-burger'],
    ['schedule-breakfast', 'item-double-smash-burger'],
    ['schedule-breakfast', 'item-cola'],
    ['schedule-breakfast', 'item-lemonade'],
    ['schedule-breakfast', 'item-iced-tea'],
    ['schedule-lunch', 'item-burger'],
    ['schedule-lunch', 'item-truffle-burger'],
    ['schedule-lunch', 'item-double-smash-burger'],
    ['schedule-lunch', 'item-fries'],
    ['schedule-lunch', 'item-cola'],
    ['schedule-lunch', 'item-lemonade'],
    ['schedule-lunch', 'item-iced-tea']
  ]) {
    db.prepare('INSERT OR IGNORE INTO menu_item_schedule (schedule_id, menu_item_id) VALUES (?, ?)').run(...mapping);
  }
}
