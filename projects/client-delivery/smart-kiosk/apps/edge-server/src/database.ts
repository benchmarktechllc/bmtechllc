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
    discount_amount_minor INTEGER NOT NULL DEFAULT 0,
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
  CREATE TABLE IF NOT EXISTS kiosk_events (
    event_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    terminal_id TEXT NOT NULL REFERENCES kiosk_terminals(terminal_id),
    event_type TEXT NOT NULL,
    context_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS kiosk_events_by_type_time ON kiosk_events(event_type, created_at);
  CREATE TABLE IF NOT EXISTS staff_audit_log (
    audit_id TEXT PRIMARY KEY,
    store_id TEXT NOT NULL REFERENCES stores(store_id),
    actor_id TEXT NOT NULL,
    action_type TEXT NOT NULL,
    target_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    details_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL
  );
  CREATE INDEX IF NOT EXISTS staff_audit_log_by_time ON staff_audit_log(created_at DESC);
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
  ensureAdditiveColumns(db);
  seedDatabase(db);
  return db;
}

function ensureAdditiveColumns(db: DatabaseSync): void {
  const orderColumns = db.prepare('PRAGMA table_info(orders)').all() as Array<{ name: string }>;
  if (!orderColumns.some((column) => column.name === 'discount_amount_minor')) {
    db.exec('ALTER TABLE orders ADD COLUMN discount_amount_minor INTEGER NOT NULL DEFAULT 0');
  }
  db.exec(`
    CREATE TABLE IF NOT EXISTS kiosk_events (
      event_id TEXT PRIMARY KEY,
      store_id TEXT NOT NULL REFERENCES stores(store_id),
      terminal_id TEXT NOT NULL REFERENCES kiosk_terminals(terminal_id),
      event_type TEXT NOT NULL,
      context_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS kiosk_events_by_type_time ON kiosk_events(event_type, created_at);
    CREATE TABLE IF NOT EXISTS staff_audit_log (
      audit_id TEXT PRIMARY KEY,
      store_id TEXT NOT NULL REFERENCES stores(store_id),
      actor_id TEXT NOT NULL,
      action_type TEXT NOT NULL,
      target_type TEXT NOT NULL,
      target_id TEXT NOT NULL,
      details_json TEXT NOT NULL DEFAULT '{}',
      created_at TEXT NOT NULL
    );
    CREATE INDEX IF NOT EXISTS staff_audit_log_by_time ON staff_audit_log(created_at DESC);
  `);
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
    ['category-burgers', 'Burgers', 1],
    ['category-sides', 'Sides', 2],
    ['category-shakes', 'Shakes', 3],
    ['category-drinks', 'Sodas', 4],
    ['category-pizzas', 'Pizzas', 5]
  ];
  for (const category of categories) {
    db.prepare(
      'INSERT OR IGNORE INTO menu_categories (category_id, category_name, display_order) VALUES (?, ?, ?)'
    ).run(...category);
  }

  for (const category of categories) {
    db.prepare('UPDATE menu_categories SET category_name = ?, display_order = ? WHERE category_id = ?').run(category[1]!, category[2]!, category[0]!);
  }

  db.prepare('UPDATE menu_items SET category_id = ? WHERE menu_item_id = ?').run('category-burgers', 'item-burger');

  const items = [
    ['item-burger', 'category-burgers', 'Classic Burger', 'LUNCH-BURGER', 825],
    ['item-truffle-burger', 'category-burgers', 'Truffle Bistro Burger', 'BURGER-TRUFFLE', 1650],
    ['item-double-smash-burger', 'category-burgers', 'Double Smash Burger', 'BURGER-DOUBLE', 1295],
    ['item-bacon-bbq-burger', 'category-burgers', 'Bacon BBQ Burger', 'BURGER-BACON-BBQ', 1095],
    ['item-jalapeno-pepper-jack-burger', 'category-burgers', 'Jalapeño Pepper Jack Burger', 'BURGER-JALAPENO-JACK', 1075],
    ['item-mushroom-swiss-burger', 'category-burgers', 'Mushroom Swiss Burger', 'BURGER-MUSHROOM-SWISS', 1125],
    ['item-fries', 'category-sides', 'Seasoned Fries', 'SIDE-FRIES', 325],
    ['item-onion-rings', 'category-sides', 'Crispy Onion Rings', 'SIDE-ONION-RINGS', 425],
    ['item-loaded-cheese-fries', 'category-sides', 'Loaded Cheese Fries', 'SIDE-LOADED-FRIES', 595],
    ['item-sweet-potato-fries', 'category-sides', 'Sweet Potato Fries', 'SIDE-SWEET-POTATO', 395],
    ['item-fried-pickles', 'category-sides', 'Fried Pickles', 'SIDE-FRIED-PICKLES', 475],
    ['item-mozzarella-sticks', 'category-sides', 'Mozzarella Sticks', 'SIDE-MOZZARELLA', 525],
    ['item-mac-cheese-bites', 'category-sides', 'Mac & Cheese Bites', 'SIDE-MAC-BITES', 495],
    ['item-chili-cheese-fries', 'category-sides', 'Chili Cheese Fries', 'SIDE-CHILI-FRIES', 625],
    ['item-garlic-parmesan-tots', 'category-sides', 'Garlic Parmesan Tots', 'SIDE-GARLIC-TOTS', 475],
    ['item-house-side-salad', 'category-sides', 'House Side Salad', 'SIDE-HOUSE-SALAD', 450],
    ['item-vanilla-bean-shake', 'category-shakes', 'Vanilla Bean Shake', 'SHAKE-VANILLA', 575],
    ['item-chocolate-malt-shake', 'category-shakes', 'Chocolate Malt Shake', 'SHAKE-CHOCOLATE-MALT', 625],
    ['item-strawberry-shake', 'category-shakes', 'Strawberry Shake', 'SHAKE-STRAWBERRY', 575],
    ['item-cookies-cream-shake', 'category-shakes', 'Cookies & Cream Shake', 'SHAKE-COOKIES-CREAM', 625],
    ['item-salted-caramel-shake', 'category-shakes', 'Salted Caramel Shake', 'SHAKE-SALTED-CARAMEL', 625],
    ['item-peanut-butter-shake', 'category-shakes', 'Peanut Butter Shake', 'SHAKE-PEANUT-BUTTER', 625],
    ['item-banana-shake', 'category-shakes', 'Banana Shake', 'SHAKE-BANANA', 575],
    ['item-coffee-shake', 'category-shakes', 'Coffee Shake', 'SHAKE-COFFEE', 625],
    ['item-oreo-espresso-shake', 'category-shakes', 'Oreo Espresso Shake', 'SHAKE-OREO-ESPRESSO', 675],
    ['item-seasonal-shake', 'category-shakes', 'Seasonal Shake', 'SHAKE-SEASONAL', 625],
    ['item-cola', 'category-drinks', 'Cola', 'DRINK-COLA', 250],
    ['item-lemonade', 'category-drinks', 'Fresh Lemonade', 'DRINK-LEMONADE', 325],
    ['item-iced-tea', 'category-drinks', 'Iced Tea', 'DRINK-TEA', 275],
    ['item-diet-cola', 'category-drinks', 'Diet Cola', 'DRINK-DIET-COLA', 250],
    ['item-root-beer', 'category-drinks', 'Root Beer', 'DRINK-ROOT-BEER', 275],
    ['item-orange-soda', 'category-drinks', 'Orange Soda', 'DRINK-ORANGE', 275],
    ['item-ginger-ale', 'category-drinks', 'Ginger Ale', 'DRINK-GINGER-ALE', 275],
    ['item-sparkling-water', 'category-drinks', 'Sparkling Water', 'DRINK-SPARKLING-WATER', 225],
    ['item-cold-brew', 'category-drinks', 'Cold Brew', 'DRINK-COLD-BREW', 350],
    ['item-bottled-water', 'category-drinks', 'Bottled Water', 'DRINK-BOTTLED-WATER', 200],
    ['item-margherita-pizza', 'category-pizzas', 'Margherita Pizza', 'PIZZA-MARGHERITA', 1195],
    ['item-pepperoni-pizza', 'category-pizzas', 'Pepperoni Pizza', 'PIZZA-PEPPERONI', 1295],
    ['item-hot-honey-pizza', 'category-pizzas', 'Hot Honey Pizza', 'PIZZA-HOT-HONEY', 1395],
    ['item-mushroom-truffle-pizza', 'category-pizzas', 'Mushroom Truffle Pizza', 'PIZZA-MUSHROOM-TRUFFLE', 1495],
    ['item-bbq-chicken-pizza', 'category-pizzas', 'BBQ Chicken Pizza', 'PIZZA-BBQ-CHICKEN', 1395],
    ['item-four-cheese-pizza', 'category-pizzas', 'Four Cheese Pizza', 'PIZZA-FOUR-CHEESE', 1295],
    ['item-veggie-supreme-pizza', 'category-pizzas', 'Veggie Supreme Pizza', 'PIZZA-VEGGIE-SUPREME', 1395],
    ['item-meat-lovers-pizza', 'category-pizzas', 'Meat Lovers Pizza', 'PIZZA-MEAT-LOVERS', 1495],
    ['item-hawaiian-pizza', 'category-pizzas', 'Hawaiian Pizza', 'PIZZA-HAWAIIAN', 1345],
    ['item-white-garlic-pizza', 'category-pizzas', 'White Garlic Pizza', 'PIZZA-WHITE-GARLIC', 1295]
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
  for (const menuItemId of [
    'item-burger',
    'item-double-smash-burger',
    'item-bacon-bbq-burger',
    'item-jalapeno-pepper-jack-burger',
    'item-mushroom-swiss-burger'
  ]) {
    db.prepare(
      'INSERT OR IGNORE INTO menu_item_modifier_groups (menu_item_id, modifier_group_id, display_order, min_selections, max_selections) VALUES (?, ?, ?, ?, ?)'
    ).run(menuItemId, 'group-truffle-size', 1, 0, 1);
  }

  const schedules = [['schedule-lunch', 'Lunch', '00:00', '23:59']];
  for (const schedule of schedules) {
    db.prepare(
      'INSERT OR IGNORE INTO menu_schedules (schedule_id, store_id, schedule_name, start_time, end_time) VALUES (?, ?, ?, ?, ?)'
    ).run(schedule[0]!, 'store-demo-001', schedule[1]!, schedule[2]!, schedule[3]!);
  }
  const allMenuItemIds = items.map(([menuItemId]) => String(menuItemId));
  for (const scheduleId of ['schedule-lunch']) {
    for (const menuItemId of allMenuItemIds) {
      db.prepare('INSERT OR IGNORE INTO menu_item_schedule (schedule_id, menu_item_id) VALUES (?, ?)').run(scheduleId, menuItemId);
    }
  }

  // Preserve historical order references while removing legacy Breakfast data
  // from all current menu and staff views on already-initialized databases.
  db.prepare('UPDATE menu_items SET is_active = 0 WHERE menu_item_id = ?').run('item-breakfast-wrap');
  db.prepare('UPDATE menu_categories SET is_active = 0 WHERE category_id = ?').run('category-breakfast');
  db.prepare('UPDATE menu_schedules SET is_active = 0 WHERE schedule_id = ?').run('schedule-breakfast');
}
