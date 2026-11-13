import { useEffect, useMemo, useState } from 'react';
import type { CartLine, OrderQuote, PaymentScenario } from '@smart-kiosk/shared';
import attractAnvilBackground from './assets/attract-anvil-background.webp';
import attractAnvilLockup from './assets/attract-anvil-lockup.svg';
import attractTouchIcon from './assets/attract-touch.svg';
import menuFigmaCardOne from './assets/optimized/menu-figma-card-1.webp';
import menuFigmaCardTwo from './assets/optimized/menu-figma-card-2.webp';
import menuFigmaCardThree from './assets/optimized/menu-figma-card-3.webp';
import menuFigmaCategoryActive from './assets/menu-figma-category-active.svg';
import menuFigmaCategoryInactive from './assets/menu-figma-category.svg';
import menuFigmaBurgerCategory from './assets/menu-figma-category-burger.svg';
import menuFigmaFriesCategory from './assets/menu-figma-category-fries.svg';
import menuFigmaCupCategory from './assets/menu-figma-category-cup.svg';
import menuFigmaCanCategory from './assets/menu-figma-category-can.svg';
import menuFigmaPizzaCategory from './assets/menu-figma-category-pizza.svg';
import menuFigmaAddBackground from './assets/menu-figma-add-bg.svg';
import menuFigmaAddIcon from './assets/menu-figma-add.svg';
import menuFigmaCartDivider from './assets/menu-figma-cart-divider.svg';
import menuFigmaCartPlus from './assets/menu-figma-cart-plus.svg';
import menuFigmaCartMinus from './assets/menu-figma-cart-minus.svg';
import menuFigmaMenuDivider from './assets/menu-figma-menu-divider.svg';
import classicBurgerImage from './assets/optimized/menu-classic-burger.webp';
import truffleBurgerImage from './assets/optimized/menu-truffle-burger.webp';
import doubleSmashBurgerImage from './assets/optimized/menu-double-smash-burger.webp';
import baconBbqBurgerImage from './assets/optimized/bacon-bbq-burger.webp';
import jalapenoPepperJackBurgerImage from './assets/optimized/jalapeno-pepper-jack-burger.webp';
import mushroomSwissBurgerImage from './assets/optimized/mushroom-swiss-burger.webp';
import friesImage from './assets/optimized/menu-fries.webp';
import shakeImage from './assets/optimized/menu-shake.webp';
import pizzaImage from './assets/optimized/menu-pizza.webp';
import colaImage from './assets/optimized/menu-cola.webp';
import lemonadeImage from './assets/optimized/menu-lemonade.webp';
import icedTeaImage from './assets/optimized/menu-iced-tea.webp';
import customizeBurgerImage from './assets/optimized/customize-burger.webp';
import customizeCloseIcon from './assets/customize-close.svg';
import customizeMinusIcon from './assets/customize-minus.svg';
import customizePlusIcon from './assets/customize-plus.svg';
import upsellSodaImage from './assets/optimized/upsell-soda.webp';
import upsellFriesImage from './assets/optimized/upsell-fries.webp';
import upsellAddIcon from './assets/upsell-add.svg';
import checkoutItemImage from './assets/optimized/checkout-item.webp';
import checkoutDivider from './assets/checkout-divider.svg';
import checkoutContactless from './assets/checkout-contactless.svg';

// Use Vite's same-origin development proxy by default. Set VITE_EDGE_URL only
// when the browser should connect directly to a separately hosted edge node.
const edgeUrl = import.meta.env.VITE_EDGE_URL ?? '';
const savedOrderKey = 'smart-kiosk:order:v1';

type Menu = {
  currencyCode: string;
  activeScheduleId: string;
  schedules: Array<{ schedule_id: string; schedule_name: string }>;
  categories: Array<{ category_id: string; category_name: string }>;
  items: Array<{
    menu_item_id: string;
    category_id: string;
    item_name: string;
    sku: string;
    base_price_minor: number;
  }>;
};

type Order = { transaction_id: string; order_status: string; total_amount_minor: number };
type MenuItem = Menu['items'][number];
type Customization = { item: MenuItem; quantity: number; size: 'single' | 'double' };
type Notice = { tone: 'info' | 'error'; message: string };
type ProcessingAction = 'checkout' | 'payment' | 'cancelling';
type StaffOverview = {
  generatedAt: string;
  integrationMode: 'mock';
  orderStatusCounts: Array<{ status: string; count: number }>;
  dispatchStatusCounts: Array<{ status: string; count: number }>;
  unavailableItems: Array<{ menu_item_id: string; item_name: string }>;
  menuItems: Array<{ menu_item_id: string; item_name: string; sku: string; base_price_minor: number; is_available: number; category_id: string; category_name: string }>;
  recentOrders: Array<{ transaction_id: string; order_status: string; total_amount_minor: number; created_at: string; item_count: number; payment_status?: string }>;
  staffIdentity: { actorId: string; role: string };
  analytics: {
    salesToday: { orderCount: number; revenueMinor: number; discountMinor: number; averageOrderMinor: number };
    paymentApprovalRate: number;
    paymentOutcomes: Array<{ status: string; count: number }>;
    topItems: Array<{ menu_item_id: string; item_name: string; quantity: number; revenue_minor: number }>;
    categorySales: Array<{ category_name: string; quantity: number; revenue_minor: number }>;
    funnel: Array<{ event_type: string; count: number }>;
  };
  deviceHealth: { network: string; paymentTerminal: string; printer: string; dispatch: string; lastActivityAt: string | null };
  auditLog: Array<{ actor_id: string; action_type: string; target_type: string; target_id: string; details_json: string; created_at: string }>;
};
type StaffMenuItem = StaffOverview['menuItems'][number];
type MenuItemDraft = { itemName: string; categoryId: string; sku: string; price: string; isAvailable: boolean };

const menuItemImages: Record<string, string> = {
  'item-burger': classicBurgerImage,
  'item-truffle-burger': truffleBurgerImage,
  'item-double-smash-burger': doubleSmashBurgerImage,
  'item-bacon-bbq-burger': baconBbqBurgerImage,
  'item-jalapeno-pepper-jack-burger': jalapenoPepperJackBurgerImage,
  'item-mushroom-swiss-burger': mushroomSwissBurgerImage,
  'item-fries': friesImage,
  'item-cola': colaImage,
  'item-lemonade': lemonadeImage,
  'item-iced-tea': icedTeaImage
};

const categoryImages: Record<string, string> = {
  'category-sides': friesImage,
  'category-shakes': shakeImage,
  'category-drinks': colaImage,
  'category-pizzas': pizzaImage
};

const menuFigmaItemImages: Record<string, string> = {
  'item-burger': menuFigmaCardOne,
  'item-truffle-burger': menuFigmaCardTwo,
  'item-double-smash-burger': menuFigmaCardThree,
};

const menuItemDescriptions: Record<string, string> = {
  'item-burger': 'Beef patty, lettuce, tomato, onion, and house sauce on a toasted bun.',
  'item-truffle-burger': 'Wagyu beef patty, wild mushroom ragout, gruyere cheese, truffle aioli.',
  'item-double-smash-burger': 'Two smashed beef patties, cheddar, pickles, onion, and burger sauce.',
  'item-bacon-bbq-burger': 'Beef patty, cheddar, crisp bacon, onion rings, and smoky barbecue sauce.',
  'item-jalapeno-pepper-jack-burger': 'Beef patty, pepper jack, jalapeños, caramelized onions, and crisp lettuce.',
  'item-mushroom-swiss-burger': 'Beef patty, melted Swiss, sautéed wild mushrooms, caramelized onions, and arugula.',
  'item-fries': 'Crisp golden fries tossed with sea salt and our signature seasoning.',
  'item-cola': 'Chilled fountain cola served over ice.',
  'item-lemonade': 'Fresh-squeezed lemonade with bright citrus and a touch of sweetness.',
  'item-iced-tea': 'Fresh-brewed iced tea with lemon.'
};

function menuImage(menuItemId: string, categoryId?: string, fallback = classicBurgerImage): string {
  return menuFigmaItemImages[menuItemId]
    ?? menuItemImages[menuItemId]
    ?? (categoryId ? categoryImages[categoryId] : undefined)
    ?? fallback;
}

const formatMoney = (amount: number, currency = 'USD') =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(amount / 100);

const paymentResultCopy: Record<PaymentScenario, string> = {
  APPROVED: 'Payment approved.',
  DECLINED: 'Your payment was declined. Please try another card or payment method.',
  CANCELLED: 'The payment was cancelled. You can try again when you are ready.',
  TIMED_OUT: 'The payment terminal timed out. Please try again.',
  FAILED: 'We could not complete that payment. Please try again or ask a team member for help.'
};

function displayOrderNumber(order: Order): string {
  return order.transaction_id.replaceAll('-', '').slice(0, 8).toUpperCase();
}

function readSavedOrder(): { cart: CartLine[]; promotionCode: string } {
  try {
    const parsed: unknown = JSON.parse(window.sessionStorage.getItem(savedOrderKey) ?? 'null');
    if (!parsed || typeof parsed !== 'object') return { cart: [], promotionCode: '' };
    const candidate = parsed as { version?: unknown; cart?: unknown; promotionCode?: unknown };
    if (candidate.version !== 1 || !Array.isArray(candidate.cart)) return { cart: [], promotionCode: '' };
    const cart = candidate.cart.filter((line): line is CartLine => (
      line !== null && typeof line === 'object' && typeof (line as CartLine).menuItemId === 'string'
      && Number.isInteger((line as CartLine).quantity) && (line as CartLine).quantity > 0
      && Array.isArray((line as CartLine).modifiers)
    ));
    return { cart, promotionCode: typeof candidate.promotionCode === 'string' ? candidate.promotionCode.slice(0, 32) : '' };
  } catch {
    return { cart: [], promotionCode: '' };
  }
}

function KioskOverlays({ notice, isIdleWarning }: { notice?: Notice; isIdleWarning: boolean }) {
  return (
    <>
      {notice && <p className={`kiosk-notice kiosk-notice--${notice.tone}`} role={notice.tone === 'error' ? 'alert' : 'status'}>{notice.message}</p>}
      {isIdleWarning && (
        <section className="kiosk-idle-warning" role="alertdialog" aria-label="Your order will reset soon">
          <p>Still choosing?</p>
          <span>Touch anywhere to continue your order.</span>
        </section>
      )}
    </>
  );
}

function StaffConsole() {
  const [accessToken, setAccessToken] = useState(() => window.sessionStorage.getItem('smart-kiosk-staff-token') ?? '');
  const [draftToken, setDraftToken] = useState(accessToken);
  const [overview, setOverview] = useState<StaffOverview>();
  const [error, setError] = useState<string>();
  const [isLoading, setIsLoading] = useState(false);
  const [refreshVersion, setRefreshVersion] = useState(0);
  const [menuPages, setMenuPages] = useState<Record<string, number>>({});
  const [selectedMenuCategoryId, setSelectedMenuCategoryId] = useState<string>();
  const [itemEditor, setItemEditor] = useState<StaffMenuItem | null | undefined>();
  const [itemDraft, setItemDraft] = useState<MenuItemDraft>({ itemName: '', categoryId: '', sku: '', price: '', isAvailable: true });

  useEffect(() => {
    if (!accessToken) return;
    let isCurrent = true;
    setIsLoading(true);
    setError(undefined);
    void fetch(`${edgeUrl}/v1/staff/overview`, { headers: { 'x-staff-token': accessToken } })
      .then(async (response) => {
        if (!response.ok) throw new Error(response.status === 401 ? 'That staff access token is not valid.' : 'Unable to load kiosk operations.');
        return (await response.json()) as StaffOverview;
      })
      .then((data) => {
        if (isCurrent) setOverview(data);
      })
      .catch((loadError: unknown) => {
        if (isCurrent) setError(loadError instanceof Error ? loadError.message : 'Unable to load kiosk operations.');
      })
      .finally(() => {
        if (isCurrent) setIsLoading(false);
      });
    return () => { isCurrent = false; };
  }, [accessToken, refreshVersion]);

  function submitAccess(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const nextToken = draftToken.trim();
    if (!nextToken) return setError('Enter a staff access token.');
    window.sessionStorage.setItem('smart-kiosk-staff-token', nextToken);
    setAccessToken(nextToken);
  }

  async function setAvailability(item: StaffOverview['menuItems'][number]) {
    setIsLoading(true);
    setError(undefined);
    try {
      const response = await fetch(`${edgeUrl}/v1/staff/menu-items/${item.menu_item_id}/availability`, {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-staff-token': accessToken },
        body: JSON.stringify({ isAvailable: item.is_available === 0 })
      });
      if (!response.ok) throw new Error('Unable to update item availability.');
      setRefreshVersion((current) => current + 1);
    } catch (updateError) {
      setError(updateError instanceof Error ? updateError.message : 'Unable to update item availability.');
      setIsLoading(false);
    }
  }

  async function cancelAwaitingOrder(transactionId: string) {
    setIsLoading(true);
    setError(undefined);
    try {
      const response = await fetch(`${edgeUrl}/v1/staff/orders/${transactionId}/cancel`, {
        method: 'POST',
        headers: { 'x-staff-token': accessToken }
      });
      if (!response.ok) throw new Error('Unable to cancel this order.');
      setRefreshVersion((current) => current + 1);
    } catch (cancelError) {
      setError(cancelError instanceof Error ? cancelError.message : 'Unable to cancel this order.');
      setIsLoading(false);
    }
  }

  const awaitingPayment = overview?.orderStatusCounts.find(({ status }) => status === 'AWAITING_PAYMENT')?.count ?? 0;
  const pendingDispatches = overview?.dispatchStatusCounts.find(({ status }) => status === 'PENDING')?.count ?? 0;
  const analytics = overview?.analytics;
  const funnelCount = (eventType: string) => analytics?.funnel.find((event) => event.event_type === eventType)?.count ?? 0;
  const menuCategories = overview?.menuItems.reduce<Array<{ categoryId: string; categoryName: string; items: StaffOverview['menuItems'] }>>((groups, item) => {
    const group = groups.find((candidate) => candidate.categoryId === item.category_id);
    if (group) group.items.push(item);
    else groups.push({ categoryId: item.category_id, categoryName: item.category_name, items: [item] });
    return groups;
  }, []) ?? [];
  const selectedMenuCategory = menuCategories.find((category) => category.categoryId === selectedMenuCategoryId) ?? menuCategories[0];

  function openNewItemEditor() {
    setItemDraft({
      itemName: '',
      categoryId: selectedMenuCategory?.categoryId ?? menuCategories[0]?.categoryId ?? '',
      sku: '',
      price: '',
      isAvailable: true
    });
    setItemEditor(null);
  }

  function openItemEditor(item: StaffMenuItem) {
    setItemDraft({
      itemName: item.item_name,
      categoryId: item.category_id,
      sku: item.sku,
      price: (item.base_price_minor / 100).toFixed(2),
      isAvailable: item.is_available === 1
    });
    setItemEditor(item);
  }

  async function saveMenuItem(event: React.FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const enteredPrice = Number(itemDraft.price);
    if (!Number.isFinite(enteredPrice) || enteredPrice < 0) return setError('Enter a valid non-negative price.');
    if (!itemDraft.categoryId) return setError('Select a category.');
    setIsLoading(true);
    setError(undefined);
    const isEditing = Boolean(itemEditor);
    const endpoint = isEditing ? `${edgeUrl}/v1/staff/menu-items/${itemEditor?.menu_item_id}` : `${edgeUrl}/v1/staff/menu-items`;
    try {
      const response = await fetch(endpoint, {
        method: isEditing ? 'PUT' : 'POST',
        headers: { 'content-type': 'application/json', 'x-staff-token': accessToken },
        body: JSON.stringify({
          categoryId: itemDraft.categoryId,
          itemName: itemDraft.itemName,
          sku: itemDraft.sku,
          basePriceMinor: Math.round(enteredPrice * 100),
          isAvailable: itemDraft.isAvailable
        })
      });
      if (!response.ok) {
        const body = await response.json().catch(() => undefined) as { message?: string } | undefined;
        throw new Error(body?.message ?? `Unable to ${isEditing ? 'update' : 'add'} menu item.`);
      }
      setSelectedMenuCategoryId(itemDraft.categoryId);
      setItemEditor(undefined);
      setRefreshVersion((current) => current + 1);
    } catch (saveError) {
      setError(saveError instanceof Error ? saveError.message : 'Unable to save menu item.');
      setIsLoading(false);
    }
  }

  if (!accessToken) {
    return (
      <main className="staff-console staff-console--access">
        <form className="staff-console__access" onSubmit={submitAccess}>
          <a href="/">← Return to kiosk</a>
          <p>Staff operations</p>
          <h1>Sign in to this kiosk</h1>
          <label htmlFor="staff-access-token">Staff access token</label>
          <input id="staff-access-token" value={draftToken} onChange={(event) => setDraftToken(event.target.value)} type="password" autoComplete="current-password" />
          {error && <p className="staff-console__error" role="alert">{error}</p>}
          <button type="submit">Open Operations</button>
        </form>
      </main>
    );
  }

  return (
    <main className="staff-console">
      <header className="staff-console__header">
        <div>
          <p>Staff operations</p>
          <h1>Kiosk control center</h1>
          <span>{overview ? `Updated ${new Date(overview.generatedAt).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}` : 'Loading status…'}</span>
        </div>
        <div>
          <button type="button" onClick={() => setRefreshVersion((current) => current + 1)} disabled={isLoading}>Refresh</button>
          <button type="button" onClick={() => { window.sessionStorage.removeItem('smart-kiosk-staff-token'); setAccessToken(''); setOverview(undefined); }}>Sign Out</button>
          <a href="/">Return to kiosk</a>
        </div>
      </header>
      {error && <p className="staff-console__error" role="alert">{error}</p>}
      <section className="staff-console__stats" aria-label="Kiosk status">
        <article><span>Integration mode</span><strong>{overview?.integrationMode ?? '—'}</strong></article>
        <article><span>Awaiting payment</span><strong>{awaitingPayment}</strong></article>
        <article><span>Pending dispatches</span><strong>{pendingDispatches}</strong></article>
        <article><span>Unavailable items</span><strong>{overview?.unavailableItems.length ?? 0}</strong></article>
      </section>
      <section className="staff-console__stats staff-console__stats--analytics" aria-label="Today's sales">
        <article><span>Sales today</span><strong>{formatMoney(analytics?.salesToday.revenueMinor ?? 0)}</strong></article>
        <article><span>Completed orders</span><strong>{analytics?.salesToday.orderCount ?? 0}</strong></article>
        <article><span>Average order</span><strong>{formatMoney(analytics?.salesToday.averageOrderMinor ?? 0)}</strong></article>
        <article><span>Payment approval</span><strong>{analytics ? `${analytics.paymentApprovalRate}%` : '—'}</strong></article>
      </section>
      <section className="staff-console__analytics-grid" aria-label="Kiosk analytics">
        <article className="staff-console__panel">
          <div className="staff-console__panel-heading"><h2>Conversion funnel</h2><span>Today’s kiosk interactions.</span></div>
          <dl className="staff-console__metric-list">
            <div><dt>Order started</dt><dd>{funnelCount('order_started')}</dd></div>
            <div><dt>Items added</dt><dd>{funnelCount('item_added')}</dd></div>
            <div><dt>Upsell viewed</dt><dd>{funnelCount('upsell_viewed')}</dd></div>
            <div><dt>Checkout viewed</dt><dd>{funnelCount('checkout_viewed')}</dd></div>
            <div><dt>Payment started</dt><dd>{funnelCount('payment_started')}</dd></div>
          </dl>
        </article>
        <article className="staff-console__panel">
          <div className="staff-console__panel-heading"><h2>Device health</h2><span>Mock adapter status until hardware is connected.</span></div>
          <dl className="staff-console__metric-list">
            <div><dt>Network</dt><dd>{overview?.deviceHealth.network ?? '—'}</dd></div>
            <div><dt>Payment terminal</dt><dd>{overview?.deviceHealth.paymentTerminal.replace('_', ' ') ?? '—'}</dd></div>
            <div><dt>Receipt printer</dt><dd>{overview?.deviceHealth.printer.replace('_', ' ') ?? '—'}</dd></div>
            <div><dt>Dispatch</dt><dd>{overview?.deviceHealth.dispatch ?? '—'}</dd></div>
          </dl>
        </article>
        <article className="staff-console__panel">
          <div className="staff-console__panel-heading"><h2>Top items</h2><span>Completed sales today.</span></div>
          <dl className="staff-console__metric-list">
            {analytics?.topItems.length ? analytics.topItems.map((item) => <div key={item.menu_item_id}><dt>{item.item_name} · {item.quantity}</dt><dd>{formatMoney(item.revenue_minor)}</dd></div>) : <p>No completed sales yet.</p>}
          </dl>
        </article>
        <article className="staff-console__panel">
          <div className="staff-console__panel-heading"><h2>Payment outcomes</h2><span>Attempts recorded today.</span></div>
          <dl className="staff-console__metric-list">
            {analytics?.paymentOutcomes.length ? analytics.paymentOutcomes.map((outcome) => <div key={outcome.status}><dt>{outcome.status.replaceAll('_', ' ')}</dt><dd>{outcome.count}</dd></div>) : <p>No payment attempts yet.</p>}
          </dl>
        </article>
      </section>
      <section className="staff-console__panel">
        <div className="staff-console__panel-heading staff-console__panel-heading--actions">
          <div><h2>Menu management</h2><span>Update item details or hide sold-out items from the kiosk.</span></div>
          <button className="staff-console__add-item" type="button" disabled={isLoading || !overview} onClick={openNewItemEditor}>Add menu item</button>
        </div>
        {overview ? <>
          <nav className="staff-console__menu-category-tabs" aria-label="Menu availability categories">
            {menuCategories.map((category) => {
              const isSelected = category.categoryId === selectedMenuCategory?.categoryId;
              return <button key={category.categoryId} type="button" className={isSelected ? 'is-selected' : ''} aria-pressed={isSelected} onClick={() => setSelectedMenuCategoryId(category.categoryId)}>{category.categoryName}</button>;
            })}
          </nav>
          {selectedMenuCategory && (() => {
          const category = selectedMenuCategory;
          const pageSize = 8;
          const pageCount = Math.max(1, Math.ceil(category.items.length / pageSize));
          const page = Math.min(menuPages[category.categoryId] ?? 0, pageCount - 1);
          const items = category.items.slice(page * pageSize, (page + 1) * pageSize);
          return (
            <section className="staff-console__menu-category" key={category.categoryId} aria-labelledby={`staff-category-${category.categoryId}`}>
              <header><h3 id={`staff-category-${category.categoryId}`}>{category.categoryName}</h3><span>{category.items.length} items</span></header>
              <div className="staff-console__menu-items">
                {items.map((item) => (
                  <article key={item.menu_item_id}>
                    <div><strong>{item.item_name}</strong><span>{item.sku} · {formatMoney(item.base_price_minor)}</span></div>
                    <div className="staff-console__item-actions">
                      <button type="button" disabled={isLoading} onClick={() => openItemEditor(item)}>Edit item</button>
                      <button type="button" disabled={isLoading} onClick={() => void setAvailability(item)}>
                        {item.is_available ? 'Mark unavailable' : 'Restore item'}
                      </button>
                    </div>
                  </article>
                ))}
              </div>
              {pageCount > 1 && <footer className="staff-console__pagination">
                <button type="button" disabled={page === 0} onClick={() => setMenuPages((current) => ({ ...current, [category.categoryId]: page - 1 }))}>Previous</button>
                <span>Page {page + 1} of {pageCount}</span>
                <button type="button" disabled={page >= pageCount - 1} onClick={() => setMenuPages((current) => ({ ...current, [category.categoryId]: page + 1 }))}>Next</button>
              </footer>}
            </section>
          );
          })()}
        </> : <p className="staff-console__empty">Loading menu items…</p>}
      </section>
      {itemEditor !== undefined && <section className="staff-console__editor-backdrop" role="presentation">
        <form className="staff-console__item-editor" onSubmit={(event) => void saveMenuItem(event)} role="dialog" aria-modal="true" aria-labelledby="staff-item-editor-title">
          <header>
            <div><p>Menu management</p><h2 id="staff-item-editor-title">{itemEditor ? 'Edit menu item' : 'Add menu item'}</h2></div>
            <button type="button" aria-label="Close item editor" disabled={isLoading} onClick={() => setItemEditor(undefined)}>×</button>
          </header>
          <div className="staff-console__editor-fields">
            <label>Item name<input required maxLength={80} value={itemDraft.itemName} onChange={(event) => setItemDraft((current) => ({ ...current, itemName: event.target.value }))} /></label>
            <label>Category<select value={itemDraft.categoryId} onChange={(event) => setItemDraft((current) => ({ ...current, categoryId: event.target.value }))}>{menuCategories.map((category) => <option key={category.categoryId} value={category.categoryId}>{category.categoryName}</option>)}</select></label>
            <label>SKU<input required pattern="[A-Za-z0-9][A-Za-z0-9-]{1,47}" title="Use letters, numbers, and hyphens." maxLength={48} value={itemDraft.sku} onChange={(event) => setItemDraft((current) => ({ ...current, sku: event.target.value.toUpperCase() }))} /></label>
            <label>Price (USD)<input required type="number" min="0" step="0.01" inputMode="decimal" value={itemDraft.price} onChange={(event) => setItemDraft((current) => ({ ...current, price: event.target.value }))} /></label>
          </div>
          <fieldset className="staff-console__availability-choice">
            <legend>Kiosk availability</legend>
            <button type="button" className={itemDraft.isAvailable ? 'is-selected' : ''} aria-pressed={itemDraft.isAvailable} onClick={() => setItemDraft((current) => ({ ...current, isAvailable: true }))}>Available now</button>
            <button type="button" className={!itemDraft.isAvailable ? 'is-selected' : ''} aria-pressed={!itemDraft.isAvailable} onClick={() => setItemDraft((current) => ({ ...current, isAvailable: false }))}>Hidden / sold out</button>
          </fieldset>
          <footer><button type="button" disabled={isLoading} onClick={() => setItemEditor(undefined)}>Cancel</button><button type="submit" disabled={isLoading}>{isLoading ? 'Saving…' : itemEditor ? 'Save changes' : 'Add item'}</button></footer>
        </form>
      </section>}
      <section className="staff-console__panel">
        <div className="staff-console__panel-heading"><h2>Staff activity</h2><span>Recent audited operations by {overview?.staffIdentity.role ?? 'staff'} role.</span></div>
        <div className="staff-console__audit-log">
          {overview?.auditLog.length ? overview.auditLog.map((entry) => (
            <article key={`${entry.created_at}-${entry.target_id}-${entry.action_type}`}>
              <strong>{entry.action_type.replaceAll('_', ' ')}</strong>
              <span>{entry.target_id}</span>
              <time dateTime={entry.created_at}>{new Date(entry.created_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</time>
            </article>
          )) : <p>No staff changes recorded yet.</p>}
        </div>
      </section>
      <section className="staff-console__panel">
        <div className="staff-console__panel-heading"><h2>Recent orders</h2><span>Newest orders are shown first.</span></div>
        <div className="staff-console__orders">
          {overview?.recentOrders.map((order) => (
            <article key={order.transaction_id}>
              <strong>#{displayOrderNumber(order)}</strong>
              <span>{order.order_status.replaceAll('_', ' ')}</span>
              <time dateTime={order.created_at}>{new Date(order.created_at).toLocaleTimeString([], { hour: 'numeric', minute: '2-digit' })}</time>
              <b>{formatMoney(order.total_amount_minor)}</b>
              {order.order_status === 'AWAITING_PAYMENT' && <button type="button" disabled={isLoading} onClick={() => void cancelAwaitingOrder(order.transaction_id)}>Cancel</button>}
            </article>
          )) ?? <p>No orders yet.</p>}
        </div>
      </section>
    </main>
  );
}

function mergeCartLine(current: CartLine[], nextLine: CartLine): CartLine[] {
  const existing = current.find((line) =>
    line.menuItemId === nextLine.menuItemId
    && line.modifiers.length === nextLine.modifiers.length
    && line.modifiers.every((modifier, index) => (
      modifier.modifierId === nextLine.modifiers[index]?.modifierId
      && modifier.quantity === nextLine.modifiers[index]?.quantity
    ))
  );
  if (!existing) return [...current, nextLine];
  return current.map((line) => (line === existing ? { ...line, quantity: line.quantity + nextLine.quantity } : line));
}

export function App() {
  return new URLSearchParams(window.location.search).has('staff') ? <StaffConsole /> : <KioskApp />;
}

function KioskApp() {
  const [isAttracting, setIsAttracting] = useState(true);
  const [menu, setMenu] = useState<Menu>();
  const [restoredOrder] = useState(readSavedOrder);
  const [cart, setCart] = useState<CartLine[]>(restoredOrder.cart);
  const [promotionCode, setPromotionCode] = useState(restoredOrder.promotionCode);
  const [quote, setQuote] = useState<OrderQuote>();
  const [activeCategory, setActiveCategory] = useState<string>();
  const [activeCategoryKey, setActiveCategoryKey] = useState('burgers');
  const [message, setMessage] = useState('Loading local menu…');
  const [notice, setNotice] = useState<Notice>();
  const [order, setOrder] = useState<Order>();
  const [confirmedOrder, setConfirmedOrder] = useState<Order>();
  const [customization, setCustomization] = useState<Customization>();
  const [isUpselling, setIsUpselling] = useState(false);
  const [isCheckingOut, setIsCheckingOut] = useState(false);
  const [isQuoting, setIsQuoting] = useState(false);
  const [isProcessing, setIsProcessing] = useState<ProcessingAction>();
  const [paymentResult, setPaymentResult] = useState<PaymentScenario>();
  const [isIdleWarning, setIsIdleWarning] = useState(false);

  useEffect(() => {
    void loadMenu();
  }, []);

  useEffect(() => {
    if (cart.length === 0) {
      window.sessionStorage.removeItem(savedOrderKey);
      return;
    }
    window.sessionStorage.setItem(savedOrderKey, JSON.stringify({ version: 1, cart, promotionCode }));
  }, [cart, promotionCode]);

  useEffect(() => {
    const onOffline = () => setNotice({ tone: 'error', message: 'Connection lost. Your order is saved on this kiosk until the connection returns.' });
    const onOnline = () => setNotice({ tone: 'info', message: 'Connection restored.' });
    window.addEventListener('offline', onOffline);
    window.addEventListener('online', onOnline);
    return () => {
      window.removeEventListener('offline', onOffline);
      window.removeEventListener('online', onOnline);
    };
  }, []);

  useEffect(() => {
    if (!notice) return undefined;
    const dismissalTimer = window.setTimeout(
      () => setNotice(undefined),
      notice.tone === 'error' ? 6_000 : 4_000
    );
    return () => window.clearTimeout(dismissalTimer);
  }, [notice]);

  useEffect(() => {
    if (cart.length === 0) {
      setQuote(undefined);
      setIsQuoting(false);
      return;
    }
    setIsQuoting(true);
    setQuote(undefined);
    void fetch(`${edgeUrl}/v1/quotes`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ items: cart, ...(promotionCode.trim() ? { promotionCode: promotionCode.trim() } : {}) })
    })
      .then(async (response) => {
        if (!response.ok) throw new Error('Unable to update total');
        return (await response.json()) as OrderQuote;
      })
      .then(setQuote)
      .catch((error: unknown) => {
        const errorMessage = error instanceof Error ? error.message : 'Unable to update total';
        setMessage(errorMessage);
        setNotice({ tone: 'error', message: errorMessage });
      })
      .finally(() => setIsQuoting(false));
  }, [cart, promotionCode]);

  useEffect(() => {
    if (isAttracting || confirmedOrder) {
      setIsIdleWarning(false);
      return undefined;
    }

    let warningTimer: number | undefined;
    let resetTimer: number | undefined;
    const armTimers = () => {
      window.clearTimeout(warningTimer);
      window.clearTimeout(resetTimer);
      warningTimer = window.setTimeout(() => setIsIdleWarning(true), 45_000);
      resetTimer = window.setTimeout(() => {
        setCart([]);
        setPromotionCode('');
        setOrder(undefined);
        setPaymentResult(undefined);
        setCustomization(undefined);
        setIsUpselling(false);
        setIsCheckingOut(false);
        setIsIdleWarning(false);
        setIsAttracting(true);
        setMessage('Order reset after inactivity');
      }, 60_000);
    };
    const recordActivity = () => {
      setIsIdleWarning(false);
      armTimers();
    };
    armTimers();
    window.addEventListener('pointerdown', recordActivity);
    window.addEventListener('keydown', recordActivity);
    return () => {
      window.clearTimeout(warningTimer);
      window.clearTimeout(resetTimer);
      window.removeEventListener('pointerdown', recordActivity);
      window.removeEventListener('keydown', recordActivity);
    };
  }, [confirmedOrder, isAttracting]);

  useEffect(() => {
    if (!customization) return undefined;
    const dismissOnEscape = (event: KeyboardEvent) => {
      if (event.key === 'Escape') setCustomization(undefined);
    };
    window.addEventListener('keydown', dismissOnEscape);
    return () => window.removeEventListener('keydown', dismissOnEscape);
  }, [customization]);

  const visibleItems = useMemo(
    () => menu?.items.filter((item) => !activeCategory || item.category_id === activeCategory) ?? [],
    [activeCategory, menu]
  );

  const categoryNavigation = useMemo(() => {
    const findCategory = (pattern: RegExp) => menu?.categories.find((category) => pattern.test(category.category_name));
    const burgerCategory = menu?.items.find((item) => /burger/i.test(item.item_name))?.category_id
      ?? findCategory(/burger|lunch/i)?.category_id
      ?? 'category-burgers';

    return [
      { key: 'burgers', categoryId: burgerCategory, icon: menuFigmaBurgerCategory, label: 'Burgers' },
      { key: 'sides', categoryId: findCategory(/side/i)?.category_id ?? 'category-sides', icon: menuFigmaFriesCategory, label: 'Sides' },
      { key: 'shakes', categoryId: findCategory(/shake/i)?.category_id ?? 'category-shakes', icon: menuFigmaCupCategory, label: 'Shakes' },
      { key: 'sodas', categoryId: findCategory(/soda|drink/i)?.category_id ?? 'category-drinks', icon: menuFigmaCanCategory, label: 'Sodas' },
      { key: 'pizzas', categoryId: findCategory(/pizza/i)?.category_id ?? 'category-pizzas', icon: menuFigmaPizzaCategory, label: 'Pizzas' }
    ];
  }, [menu]);

  const activeCategoryLabel = categoryNavigation.find((category) => category.key === activeCategoryKey)?.label
    ?? menu?.categories.find((category) => category.category_id === activeCategory)?.category_name
    ?? 'Menu';

  const cartEntries = useMemo(() => cart.map((line, index) => {
    const item = menu?.items.find((candidate) => candidate.menu_item_id === line.menuItemId);
    const pricedLine = quote?.lines[index];
    return {
      line,
      index,
      itemName: pricedLine?.name ?? item?.item_name ?? 'Selected item',
      modifiers: pricedLine?.modifiers.map((modifier) => modifier.name) ?? [],
      total: pricedLine?.lineTotalMinor ?? (item?.base_price_minor ?? 0) * line.quantity
    };
  }), [cart, menu, quote]);

  async function loadMenu(scheduleId?: string) {
    try {
      const response = await fetch(`${edgeUrl}/v1/menu${scheduleId ? `?scheduleId=${scheduleId}` : ''}`);
      if (!response.ok) throw new Error('Local edge service is unavailable');
      const data = (await response.json()) as Menu;
      setMenu(data);
      setActiveCategory(
        data.items.find((item) => /burger/i.test(item.item_name))?.category_id
          ?? data.categories.find((category) => /burger|lunch/i.test(category.category_name))?.category_id
          ?? 'category-burgers'
      );
      setActiveCategoryKey('burgers');
      setIsUpselling(false);
      setIsCheckingOut(false);
      setConfirmedOrder(undefined);
      setPaymentResult(undefined);
      setMessage(`Showing ${data.activeScheduleId.replace('schedule-', '')} menu`);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Local edge service is unavailable';
      setMessage(errorMessage);
      setNotice({ tone: 'error', message: errorMessage });
    }
  }

  function customizationLine({ item, quantity, size }: Customization): CartLine {
    const modifiers = item.category_id === 'category-burgers' && size === 'double'
      ? [{ modifierId: 'modifier-truffle-double', quantity: 1 }]
      : [];
    return { menuItemId: item.menu_item_id, quantity, modifiers };
  }

  function openCustomization(item: MenuItem) {
    setCustomization({
      item,
      quantity: 1,
      size: item.menu_item_id === 'item-truffle-burger' ? 'double' : 'single'
    });
  }

  function recordKioskEvent(eventType: 'order_started' | 'item_added' | 'upsell_viewed' | 'upsell_added' | 'checkout_viewed' | 'payment_started', context: Record<string, unknown> = {}) {
    void fetch(`${edgeUrl}/v1/telemetry/events`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ eventType, context })
    }).catch(() => undefined);
  }

  function addCustomization() {
    if (!customization) return;
    const line = customizationLine(customization);
    setCart((current) => mergeCartLine(current, line));
    setMessage('Added to cart');
    setNotice({ tone: 'info', message: `${customization.item.item_name} added to your order.` });
    recordKioskEvent('item_added', { menuItemId: customization.item.menu_item_id, quantity: customization.quantity });
    setCustomization(undefined);
  }

  function changeCartQuantity(index: number, amount: number) {
    setCart((current) => current.flatMap((line, lineIndex) => {
      if (lineIndex !== index) return [line];
      const quantity = line.quantity + amount;
      return quantity > 0 ? [{ ...line, quantity }] : [];
    }));
    setOrder(undefined);
    setPaymentResult(undefined);
  }

  function addUpsellItem(menuItemId: string, itemName: string) {
    setCart((current) => mergeCartLine(current, { menuItemId, quantity: 1, modifiers: [] }));
    setMessage(`${itemName} added to cart`);
    setNotice({ tone: 'info', message: `${itemName} added to your order.` });
    recordKioskEvent('item_added', { menuItemId });
    recordKioskEvent('upsell_added', { menuItemId });
  }

  function continueToCheckout() {
    setIsUpselling(false);
    setIsCheckingOut(true);
    setPaymentResult(undefined);
    setNotice(undefined);
    recordKioskEvent('checkout_viewed');
  }

  async function cancelCheckout() {
    if (!order) {
      setCart([]);
      setPromotionCode('');
      setIsCheckingOut(false);
      setMessage('Order cancelled');
      setNotice({ tone: 'info', message: 'Your order was cancelled.' });
      return;
    }

    setIsProcessing('cancelling');
    try {
      const response = await fetch(`${edgeUrl}/v1/orders/${order.transaction_id}/cancel`, { method: 'POST' });
      if (!response.ok) throw new Error('Unable to cancel this order. Please ask a team member for help.');
      setCart([]);
      setPromotionCode('');
      setOrder(undefined);
      setPaymentResult(undefined);
      setIsCheckingOut(false);
      setMessage('Order cancelled');
      setNotice({ tone: 'info', message: 'Your order was cancelled.' });
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Unable to cancel this order.';
      setMessage(errorMessage);
      setNotice({ tone: 'error', message: errorMessage });
    } finally {
      setIsProcessing(undefined);
    }
  }

  async function checkout(items = cart) {
    if (items.length === 0) return;
    setIsProcessing('checkout');
    setNotice(undefined);
    recordKioskEvent('payment_started', { itemCount: items.reduce((sum, item) => sum + item.quantity, 0) });
    try {
      const response = await fetch(`${edgeUrl}/v1/orders`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({
          items,
          ...(promotionCode.trim() ? { promotionCode: promotionCode.trim() } : {}),
          idempotencyKey: crypto.randomUUID(),
          orderType: 'TAKE_OUT'
        })
      });
      if (!response.ok) throw new Error('Order could not be created. Review your cart and try again.');
      const created = (await response.json()) as Order;
      setOrder(created);
      setPaymentResult(undefined);
      setIsUpselling(false);
      setMessage('Order created. Choose a simulated payment outcome.');
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Order could not be created.';
      setMessage(errorMessage);
      setNotice({ tone: 'error', message: errorMessage });
    } finally {
      setIsProcessing(undefined);
    }
  }

  async function pay(scenario: PaymentScenario) {
    if (!order) return;
    setIsProcessing('payment');
    try {
      const response = await fetch(`${edgeUrl}/v1/orders/${order.transaction_id}/payment`, {
        method: 'POST',
        headers: { 'content-type': 'application/json' },
        body: JSON.stringify({ scenario })
      });
      if (!response.ok) throw new Error('Payment could not be completed. Your cart remains available.');
      const updated = (await response.json()) as Order;
      setOrder(updated);
      if (updated.order_status === 'COMPLETED') {
        setMessage('Simulated payment approved. Mock kitchen and POS dispatches acknowledged.');
        setCart([]);
        setPromotionCode('');
        setOrder(undefined);
        setIsCheckingOut(false);
        setConfirmedOrder(updated);
      } else {
        setMessage(`Simulated payment ${scenario.toLowerCase().replace('_', ' ')}. Your cart remains available.`);
        setPaymentResult(scenario);
      }
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : 'Payment could not be completed. Your cart remains available.';
      setMessage(errorMessage);
      setNotice({ tone: 'error', message: errorMessage });
    } finally {
      setIsProcessing(undefined);
    }
  }

  function startNewOrder() {
    setConfirmedOrder(undefined);
    setOrder(undefined);
    setPaymentResult(undefined);
    setCustomization(undefined);
    setIsUpselling(false);
    setIsCheckingOut(false);
    setCart([]);
    setPromotionCode('');
    setIsAttracting(true);
    setNotice(undefined);
    setMessage('Ready for a new order');
  }

  if (isAttracting) {
    return (
      <main className="attract-screen">
        <KioskOverlays notice={notice} isIdleWarning={isIdleWarning} />
        <button
          type="button"
          className="attract-screen__start"
          onClick={() => { recordKioskEvent('order_started'); setIsAttracting(false); }}
          aria-label="Start your order"
        >
          <img className="attract-screen__background" src={attractAnvilBackground} alt="" />
          <span className="attract-screen__shade" aria-hidden="true" />
          <img className="attract-screen__lockup" src={attractAnvilLockup} alt="" aria-hidden="true" />
          <span className="attract-screen__prompt" aria-hidden="true">
            <img src={attractTouchIcon} alt="" />
            <span>Touch Screen To Start Ordering</span>
          </span>
        </button>
      </main>
    );
  }

  if (confirmedOrder) {
    return (
      <main className="kiosk-completed-screen" aria-labelledby="completed-order-title">
        <KioskOverlays notice={notice} isIdleWarning={isIdleWarning} />
        <section className="kiosk-completed">
          <h1 id="completed-order-title">Order #{displayOrderNumber(confirmedOrder)}</h1>
          <p>Your receipt is printing.</p>
          <button type="button" onClick={startNewOrder}>Start New Order</button>
          <span className="sr-only" role="status">Order {confirmedOrder.transaction_id} completed successfully.</span>
        </section>
      </main>
    );
  }

  if (isCheckingOut) {
    const subtotal = quote?.subtotalMinor ?? 0;
    const discount = quote?.discountAmountMinor ?? 0;
    const tax = quote?.taxAmountMinor ?? 0;
    const total = quote?.totalAmountMinor ?? 0;

    return (
      <main className="kiosk-checkout-screen">
        <KioskOverlays notice={notice} isIdleWarning={isIdleWarning} />
        <section className="kiosk-checkout-summary" aria-labelledby="order-summary-title">
          <header>
            <h1 id="order-summary-title">Order Summary</h1>
            <p>Ticket #10005</p>
          </header>
          <div className="kiosk-checkout-summary__items">
            {cartEntries.map(({ line, index, itemName, modifiers, total: lineTotal }) => (
              <article className="kiosk-checkout-summary__item" key={`${line.menuItemId}-${index}`}>
                <img src={menuImage(line.menuItemId, menu?.items.find((item) => item.menu_item_id === line.menuItemId)?.category_id, checkoutItemImage)} alt="" />
                <div>
                  <h2>{itemName}</h2>
                  <p>{modifiers.join(', ') || 'Standard'}</p>
                  <strong>{formatMoney(lineTotal, menu?.currencyCode)}</strong>
                </div>
              </article>
            ))}
          </div>
          <footer className="kiosk-checkout-summary__totals">
            <img src={checkoutDivider} alt="" />
            <dl>
              <div><dt>Subtotal</dt><dd>{formatMoney(subtotal, menu?.currencyCode)}</dd></div>
              {discount > 0 && <div><dt>Promotion</dt><dd>−{formatMoney(discount, menu?.currencyCode)}</dd></div>}
              <div><dt>Tax</dt><dd>{formatMoney(tax, menu?.currencyCode)}</dd></div>
              <div className="kiosk-checkout-summary__grand-total"><dt>Total</dt><dd>{formatMoney(total, menu?.currencyCode)}</dd></div>
            </dl>
          </footer>
        </section>
        <section className="kiosk-checkout-payment" aria-labelledby="payment-title">
          {!order ? (
            <button
              type="button"
              className="kiosk-checkout-payment__contactless"
              onClick={() => void checkout()}
              aria-label="Start payment"
              disabled={isProcessing === 'checkout' || isQuoting}
            >
              <img src={checkoutContactless} alt="" />
            </button>
          ) : paymentResult ? (
            <div className="kiosk-checkout-payment__result" role="status">
              <h2>Payment not completed</h2>
              <p>{paymentResultCopy[paymentResult]}</p>
              <button type="button" onClick={() => setPaymentResult(undefined)}>Try Another Payment</button>
            </div>
          ) : (
            <div className="kiosk-checkout-payment__scenarios">
              <span>Test terminal response</span>
              {(['APPROVED', 'DECLINED', 'CANCELLED', 'TIMED_OUT', 'FAILED'] as PaymentScenario[]).map((scenario) => (
                <button type="button" key={scenario} disabled={isProcessing === 'payment'} onClick={() => void pay(scenario)}>{scenario.replace('_', ' ')}</button>
              ))}
            </div>
          )}
          <h1 id="payment-title">Payment</h1>
          <p>{isProcessing === 'checkout' ? 'Starting secure payment…' : 'Tap, insert, or swipe your preferred method below.'}</p>
          <span className="kiosk-checkout-payment__secure">Secure Encrypted Transaction</span>
          <button type="button" className="kiosk-checkout-payment__cancel" disabled={Boolean(isProcessing)} onClick={() => void cancelCheckout()}>
            {isProcessing === 'cancelling' ? 'Cancelling Order…' : 'Cancel Order'}
          </button>
        </section>
      </main>
    );
  }

  if (isUpselling) {
    const upsellItems = [
      { menuItemId: 'item-cola', name: 'Large Soda', image: upsellSodaImage, alt: 'Large fountain soda' },
      { menuItemId: 'item-fries', name: 'Fries', image: upsellFriesImage, alt: 'French fries' }
    ];

    return (
      <main className="kiosk-upsell-screen">
        <KioskOverlays notice={notice} isIdleWarning={isIdleWarning} />
        <section className="kiosk-upsell" aria-labelledby="upsell-title">
          <h1 id="upsell-title">Complete Your Meal?</h1>
          <p>Perfect pairings to elevate your dining experience.</p>
          <div className="kiosk-upsell__grid">
            {upsellItems.map((item) => (
              <article className="kiosk-upsell__card" key={item.menuItemId}>
                <img className="kiosk-upsell__image" src={item.image} alt={item.alt} />
                <div className="kiosk-upsell__card-content">
                  <div>
                    <h2>{item.name}</h2>
                    <span>{formatMoney(menu?.items.find((menuItem) => menuItem.menu_item_id === item.menuItemId)?.base_price_minor ?? 0, menu?.currencyCode)}</span>
                  </div>
                  <button type="button" onClick={() => addUpsellItem(item.menuItemId, item.name)}>
                    <img src={upsellAddIcon} alt="" />
                    Add
                  </button>
                </div>
              </article>
            ))}
          </div>
          <button type="button" className="kiosk-upsell__skip" onClick={continueToCheckout}>
            No Thanks, Continue to Checkout
          </button>
        </section>
        {cart.length > 0 && (
          <button type="button" className="kiosk-order-peek" onClick={() => setIsUpselling(false)}>
            View Order · {cart.reduce((sum, line) => sum + line.quantity, 0)} items · {formatMoney(quote?.totalAmountMinor ?? 0, menu?.currencyCode)}
          </button>
        )}
      </main>
    );
  }

  return (
    <main className="kiosk-menu-screen kiosk-menu-screen--figma">
      <KioskOverlays notice={notice} isIdleWarning={isIdleWarning} />
      <aside className="kiosk-category-nav" aria-label="Menu categories">
        {categoryNavigation.map((category) => {
          const isActive = category.key === activeCategoryKey;
          return (
            <button
              key={category.key}
              type="button"
              className={`kiosk-category-nav__button${isActive ? ' kiosk-category-nav__button--active' : ''}`}
              aria-pressed={isActive}
              aria-label={category.label}
              onClick={() => {
                setActiveCategory(category.categoryId);
                setActiveCategoryKey(category.key);
              }}
            >
              <span className="kiosk-category-nav__icon" aria-hidden="true">
                <img src={isActive ? menuFigmaCategoryActive : menuFigmaCategoryInactive} alt="" />
                {!isActive && category.key === 'burgers' ? (
                  <span className="kiosk-category-nav__symbol kiosk-category-nav__symbol--burgers" />
                ) : (
                  <img className={`kiosk-category-nav__symbol kiosk-category-nav__symbol--${category.key}${isActive ? ' is-active' : ''}`} src={category.icon} alt="" />
                )}
              </span>
              <span className="kiosk-category-nav__label">{category.label}</span>
            </button>
          );
        })}
        <img className="kiosk-menu-screen__divider" src={menuFigmaMenuDivider} alt="" aria-hidden="true" />
      </aside>
      <section className="kiosk-menu-content" aria-label={`${activeCategoryLabel} menu`}>
        <p className="sr-only" role="status">{message}</p>
        {visibleItems.length > 0 ? (
          <div className="kiosk-menu-grid">
            {visibleItems.map((item) => (
              <article className="kiosk-menu-card" key={item.menu_item_id}>
                <img className="kiosk-menu-card__image" src={menuImage(item.menu_item_id, item.category_id)} alt={item.item_name} loading="lazy" decoding="async" />
                <div className="kiosk-menu-card__content">
                  <h2>{item.item_name}</h2>
                  <strong>{formatMoney(item.base_price_minor, menu?.currencyCode)}</strong>
                  <button type="button" aria-label={`Customize ${item.item_name}`} onClick={() => openCustomization(item)}>
                    <img src={menuFigmaAddBackground} alt="" />
                    <img src={menuFigmaAddIcon} alt="" />
                  </button>
                </div>
              </article>
            ))}
          </div>
        ) : (
          <p className="kiosk-menu-empty">No items are available in this category.</p>
        )}
      </section>
      <aside className="kiosk-cart" aria-label="Your order">
        <img className="kiosk-cart__brand" src={attractAnvilLockup} alt="ANVIL" />
        <header className="kiosk-cart__header">
          <h2>Your Order</h2>
          <span>{cart.reduce((sum, line) => sum + line.quantity, 0)} items</span>
        </header>
        <img className="kiosk-cart__divider" src={menuFigmaCartDivider} alt="" aria-hidden="true" />
        {cartEntries.length > 0 ? (
          <ul className="kiosk-cart__items">
            {cartEntries.map(({ line, index, itemName, modifiers, total }) => (
              <li key={`${line.menuItemId}-${index}`} className="kiosk-cart__item">
                <div>
                  <strong>{itemName}</strong>
                  {modifiers.length > 0 && <small>{modifiers.join(', ')}</small>}
                  <span>{formatMoney(total, menu?.currencyCode)}</span>
                </div>
                <div className="kiosk-cart__quantity" aria-label={`Quantity for ${itemName}`}>
                  <button type="button" onClick={() => changeCartQuantity(index, -1)} aria-label={`Remove one ${itemName}`}><img src={menuFigmaCartMinus} alt="" /></button>
                  <span>{line.quantity}</span>
                  <button type="button" onClick={() => changeCartQuantity(index, 1)} aria-label={`Add one ${itemName}`}><img src={menuFigmaCartPlus} alt="" /></button>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="kiosk-cart__empty">Your order is empty. Choose an item to get started.</p>
        )}
        <footer className="kiosk-cart__footer">
          <img className="kiosk-cart__divider" src={menuFigmaCartDivider} alt="" aria-hidden="true" />
          <div className="kiosk-cart__total">
            <span>Total</span>
            <strong>{isQuoting ? 'Updating…' : formatMoney(quote?.totalAmountMinor ?? 0, menu?.currencyCode)}</strong>
          </div>
          {!order ? (
            <button type="button" disabled={!quote || isQuoting} onClick={() => { recordKioskEvent('upsell_viewed'); setIsUpselling(true); }}>
              {isQuoting ? 'Updating Order…' : 'Checkout'}
            </button>
          ) : (
            <div className="kiosk-cart__payment">
              <span>Simulated payment</span>
              {(['APPROVED', 'DECLINED', 'CANCELLED', 'TIMED_OUT', 'FAILED'] as PaymentScenario[]).map((scenario) => (
                <button type="button" key={scenario} onClick={() => void pay(scenario)}>{scenario.replace('_', ' ')}</button>
              ))}
            </div>
          )}
        </footer>
      </aside>
      {customization && (
        <section
          className="kiosk-customization-modal"
          role="presentation"
          onMouseDown={(event) => {
            if (event.target === event.currentTarget) setCustomization(undefined);
          }}
        >
          <div className="kiosk-customization kiosk-customization--modal" role="dialog" aria-modal="true" aria-labelledby="customization-title">
            <header className="kiosk-customization__modal-header">
              <div>
                <p>Make it yours</p>
                <h1 id="customization-title">Customize your order</h1>
              </div>
              <button type="button" className="kiosk-customization__modal-close" onClick={() => setCustomization(undefined)} aria-label="Close customization">
                <img src={customizeCloseIcon} alt="" />
              </button>
            </header>
            <div className="kiosk-customization__layout">
              <article className="kiosk-customization__item-card">
                <img
                  className="kiosk-customization__item-image"
                  src={menuImage(customization.item.menu_item_id, customization.item.category_id, customizeBurgerImage)}
                  alt={customization.item.item_name}
                />
                <div className="kiosk-customization__item-copy">
                  <h2>{customization.item.item_name}</h2>
                  <p>{menuItemDescriptions[customization.item.menu_item_id] ?? 'Made fresh to order.'}</p>
                  <span>1150 cal</span>
                </div>
              </article>
              <section className="kiosk-customization__controls" aria-label="Customization options">
                <div className="kiosk-customization__amount-row">
                  <h2>Amount</h2>
                  <div className="kiosk-quantity-control" aria-label="Item amount">
                    <button type="button" disabled={customization.quantity <= 1} onClick={() => setCustomization((current) => current ? { ...current, quantity: Math.max(1, current.quantity - 1) } : current)} aria-label="Decrease amount">
                      <img src={customizeMinusIcon} alt="" />
                    </button>
                    <span>{customization.quantity}</span>
                    <button type="button" onClick={() => setCustomization((current) => current ? { ...current, quantity: Math.min(20, current.quantity + 1) } : current)} aria-label="Increase amount">
                      <span className="kiosk-quantity-control__plus"><img src={customizePlusIcon} alt="" /><img src={customizePlusIcon} alt="" /></span>
                    </button>
                  </div>
                </div>
                {customization.item.category_id === 'category-burgers' && (
                  <section className="kiosk-customization__size-section" aria-labelledby="burger-size-title">
                    <h2 id="burger-size-title">Burger size</h2>
                    <div className="kiosk-customization__size-options">
                      <button type="button" className={customization.size === 'single' ? 'is-selected' : ''} aria-pressed={customization.size === 'single'} onClick={() => setCustomization((current) => current ? { ...current, size: 'single' } : current)}>
                        <span>Single</span><small>1 patty</small>
                      </button>
                      <button type="button" className={customization.size === 'double' ? 'is-selected' : ''} aria-pressed={customization.size === 'double'} onClick={() => setCustomization((current) => current ? { ...current, size: 'double' } : current)}>
                        <span>Double</span><small>2 patties</small>
                      </button>
                    </div>
                  </section>
                )}
                <footer className="kiosk-customization__actions">
                  <button type="button" className="kiosk-customization__add-more" onClick={addCustomization}>Add to Order</button>
                </footer>
              </section>
            </div>
          </div>
        </section>
      )}
    </main>
  );
}
