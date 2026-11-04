import { useEffect, useMemo, useState } from 'react';
import type { CartLine, OrderQuote, PaymentScenario } from '@smart-kiosk/shared';
import attractBackground from './assets/attract-background.webp';
import breakfastWrapImage from './assets/menu-breakfast-wrap.jpg';
import classicBurgerImage from './assets/menu-classic-burger.jpg';
import truffleBurgerImage from './assets/menu-truffle-burger.webp';
import doubleSmashBurgerImage from './assets/menu-double-smash-burger.webp';
import friesImage from './assets/menu-fries.jpg';
import colaImage from './assets/menu-cola.webp';
import lemonadeImage from './assets/menu-lemonade.jpg';
import icedTeaImage from './assets/menu-iced-tea.webp';
import menuAddIcon from './assets/menu-icon-add.svg';
import menuBurgerIcon from './assets/menu-icon-burger.svg';
import menuDrinkIcon from './assets/menu-icon-drink.svg';
import menuFriesIcon from './assets/menu-icon-fries.svg';
import customizeBurgerImage from './assets/customize-burger.webp';
import customizeCloseIcon from './assets/customize-close.svg';
import customizeMinusIcon from './assets/customize-minus.svg';
import customizePlusIcon from './assets/customize-plus.svg';
import upsellSodaImage from './assets/upsell-soda.webp';
import upsellFriesImage from './assets/upsell-fries.webp';
import upsellAddIcon from './assets/upsell-add.svg';
import checkoutItemImage from './assets/checkout-item.webp';
import checkoutDivider from './assets/checkout-divider.svg';
import checkoutContactless from './assets/checkout-contactless.svg';

// Use Vite's same-origin development proxy by default. Set VITE_EDGE_URL only
// when the browser should connect directly to a separately hosted edge node.
const edgeUrl = import.meta.env.VITE_EDGE_URL ?? '';

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

const menuItemImages: Record<string, string> = {
  'item-breakfast-wrap': breakfastWrapImage,
  'item-burger': classicBurgerImage,
  'item-truffle-burger': truffleBurgerImage,
  'item-double-smash-burger': doubleSmashBurgerImage,
  'item-fries': friesImage,
  'item-cola': colaImage,
  'item-lemonade': lemonadeImage,
  'item-iced-tea': icedTeaImage
};

const menuItemDescriptions: Record<string, string> = {
  'item-breakfast-wrap': 'Eggs, cheddar, roasted peppers, and potatoes wrapped in a warm tortilla.',
  'item-burger': 'Beef patty, lettuce, tomato, onion, and house sauce on a toasted bun.',
  'item-truffle-burger': 'Wagyu beef patty, wild mushroom ragout, gruyere cheese, truffle aioli.',
  'item-double-smash-burger': 'Two smashed beef patties, cheddar, pickles, onion, and burger sauce.',
  'item-fries': 'Crisp golden fries tossed with sea salt and our signature seasoning.',
  'item-cola': 'Chilled fountain cola served over ice.',
  'item-lemonade': 'Fresh-squeezed lemonade with bright citrus and a touch of sweetness.',
  'item-iced-tea': 'Fresh-brewed iced tea with lemon.'
};

const formatMoney = (amount: number, currency = 'USD') =>
  new Intl.NumberFormat('en-US', { style: 'currency', currency }).format(amount / 100);

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
  const [isAttracting, setIsAttracting] = useState(true);
  const [menu, setMenu] = useState<Menu>();
  const [cart, setCart] = useState<CartLine[]>([]);
  const [quote, setQuote] = useState<OrderQuote>();
  const [activeCategory, setActiveCategory] = useState<string>();
  const [message, setMessage] = useState('Loading local menu…');
  const [order, setOrder] = useState<Order>();
  const [confirmedOrder, setConfirmedOrder] = useState<Order>();
  const [customization, setCustomization] = useState<Customization>();
  const [isUpselling, setIsUpselling] = useState(false);
  const [isCheckingOut, setIsCheckingOut] = useState(false);

  useEffect(() => {
    void loadMenu();
  }, []);

  useEffect(() => {
    if (cart.length === 0) {
      setQuote(undefined);
      return;
    }
    void fetch(`${edgeUrl}/v1/quotes`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ items: cart })
    })
      .then(async (response) => {
        if (!response.ok) throw new Error('Unable to update total');
        return (await response.json()) as OrderQuote;
      })
      .then(setQuote)
      .catch((error: unknown) => setMessage(error instanceof Error ? error.message : 'Unable to update total'));
  }, [cart]);

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
      { categoryId: burgerCategory, icon: menuBurgerIcon, label: 'Burgers' },
      { categoryId: findCategory(/side/i)?.category_id ?? 'category-sides', icon: menuFriesIcon, label: 'Sides' },
      { categoryId: findCategory(/drink/i)?.category_id ?? 'category-drinks', icon: menuDrinkIcon, label: 'Drinks' }
    ];
  }, [menu]);

  const displayItems = useMemo(() => {
    if (visibleItems.length === 0) return [];
    return Array.from({ length: Math.max(visibleItems.length, 3) }, (_, index) => visibleItems[index % visibleItems.length]!);
  }, [visibleItems]);

  const activeCategoryLabel = categoryNavigation.find((category) => category.categoryId === activeCategory)?.label
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
      setCart([]);
      setIsUpselling(false);
      setIsCheckingOut(false);
      setConfirmedOrder(undefined);
      setMessage(`Showing ${data.activeScheduleId.replace('schedule-', '')} menu`);
    } catch (error) {
      setMessage(error instanceof Error ? error.message : 'Local edge service is unavailable');
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

  function addCustomization() {
    if (!customization) return;
    const line = customizationLine(customization);
    setCart((current) => mergeCartLine(current, line));
    setMessage('Added to cart');
    setCustomization(undefined);
  }

  function changeCartQuantity(index: number, amount: number) {
    setCart((current) => current.flatMap((line, lineIndex) => {
      if (lineIndex !== index) return [line];
      const quantity = line.quantity + amount;
      return quantity > 0 ? [{ ...line, quantity }] : [];
    }));
    setOrder(undefined);
  }

  function addUpsellItem(menuItemId: string, itemName: string) {
    setCart((current) => mergeCartLine(current, { menuItemId, quantity: 1, modifiers: [] }));
    setMessage(`${itemName} added to cart`);
  }

  function continueToCheckout() {
    setIsUpselling(false);
    setIsCheckingOut(true);
  }

  function cancelCheckout() {
    if (order) {
      setCart([]);
      void pay('CANCELLED');
      return;
    }
    setCart([]);
    setIsCheckingOut(false);
    setMessage('Order cancelled');
  }

  async function checkout(items = cart) {
    if (items.length === 0) return;
    const response = await fetch(`${edgeUrl}/v1/orders`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ items, idempotencyKey: crypto.randomUUID(), orderType: 'TAKE_OUT' })
    });
    if (!response.ok) return setMessage('Order could not be created. Review your cart and try again.');
    const created = (await response.json()) as Order;
    setOrder(created);
    setIsUpselling(false);
    setMessage('Order created. Choose a simulated payment outcome.');
  }

  async function pay(scenario: PaymentScenario) {
    if (!order) return;
    const response = await fetch(`${edgeUrl}/v1/orders/${order.transaction_id}/payment`, {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ scenario })
    });
    if (!response.ok) return setMessage('Payment could not be completed. Your cart remains available.');
    const updated = (await response.json()) as Order;
    setOrder(updated);
    if (updated.order_status === 'COMPLETED') {
      setMessage('Simulated payment approved. Mock kitchen and POS dispatches acknowledged.');
      setCart([]);
      setOrder(undefined);
      setIsCheckingOut(false);
      setConfirmedOrder(updated);
    } else {
      setMessage(`Simulated payment ${scenario.toLowerCase().replace('_', ' ')}. Your cart remains available.`);
      setOrder(undefined);
      setIsCheckingOut(false);
    }
  }

  function startNewOrder() {
    setConfirmedOrder(undefined);
    setOrder(undefined);
    setCustomization(undefined);
    setIsUpselling(false);
    setIsCheckingOut(false);
    setCart([]);
    setIsAttracting(true);
    setMessage('Ready for a new order');
  }

  if (isAttracting) {
    return (
      <main className="attract-screen">
        <button
          type="button"
          className="attract-screen__start"
          onClick={() => setIsAttracting(false)}
          aria-label="Start your order"
        >
          <img className="attract-screen__background" src={attractBackground} alt="" />
          <span className="attract-screen__shade" aria-hidden="true" />
          <span className="attract-screen__prompt">Tap Screen Start Order</span>
        </button>
      </main>
    );
  }

  if (confirmedOrder) {
    return (
      <main className="kiosk-completed-screen" aria-labelledby="completed-order-title">
        <section className="kiosk-completed">
          <h1 id="completed-order-title">Order #10005</h1>
          <p>Your receipt is printing.</p>
          <button type="button" onClick={startNewOrder}>Start New Order</button>
          <span className="sr-only" role="status">Order {confirmedOrder.transaction_id} completed successfully.</span>
        </section>
      </main>
    );
  }

  if (customization) {
    const hasSizeOptions = customization.item.category_id === 'category-burgers';
    const customizationImage = customization.item.menu_item_id === 'item-truffle-burger'
      ? customizeBurgerImage
      : menuItemImages[customization.item.menu_item_id] ?? customizeBurgerImage;

    return (
      <main className="kiosk-customization-screen">
        <div className="kiosk-customization">
          <h1>Customize Order</h1>
          <div className="kiosk-customization__layout">
            <article className="kiosk-customization__item-card">
              <img className="kiosk-customization__item-image" src={customizationImage} alt={customization.item.item_name} />
              <button
                type="button"
                className="kiosk-customization__close"
                onClick={() => setCustomization(undefined)}
                aria-label="Return to menu"
              >
                <img src={customizeCloseIcon} alt="" />
              </button>
              <div className="kiosk-customization__item-copy">
                <h2>{customization.item.item_name}</h2>
                <p>{menuItemDescriptions[customization.item.menu_item_id] ?? 'Made fresh to order.'}</p>
                <span>1150 cal</span>
              </div>
            </article>
            <section className="kiosk-customization__controls" aria-label={`Customize ${customization.item.item_name}`}>
              <div className="kiosk-customization__amount-row">
                <h2>Amount</h2>
                <div className="kiosk-quantity-control" aria-label="Item quantity">
                  <button
                    type="button"
                    onClick={() => setCustomization((current) => current && { ...current, quantity: Math.max(1, current.quantity - 1) })}
                    aria-label="Decrease quantity"
                    disabled={customization.quantity === 1}
                  >
                    <img src={customizeMinusIcon} alt="" />
                  </button>
                  <span>{customization.quantity}</span>
                  <button
                    type="button"
                    onClick={() => setCustomization((current) => current && { ...current, quantity: Math.min(20, current.quantity + 1) })}
                    aria-label="Increase quantity"
                  >
                    <span className="kiosk-quantity-control__plus" aria-hidden="true">
                      <img src={customizePlusIcon} alt="" />
                      <img src={customizePlusIcon} alt="" />
                    </span>
                  </button>
                </div>
              </div>
              {hasSizeOptions && (
                <div className="kiosk-customization__size-section">
                  <h2>Size <em>*</em></h2>
                  <div className="kiosk-customization__size-options">
                    <button
                      type="button"
                      className={customization.size === 'single' ? 'is-selected' : ''}
                      onClick={() => setCustomization((current) => current && { ...current, size: 'single' })}
                      aria-pressed={customization.size === 'single'}
                    >
                      <span>Single</span>
                    </button>
                    <button
                      type="button"
                      className={customization.size === 'double' ? 'is-selected' : ''}
                      onClick={() => setCustomization((current) => current && { ...current, size: 'double' })}
                      aria-pressed={customization.size === 'double'}
                    >
                      <span>Double</span>
                      <small>+ $4.00</small>
                    </button>
                  </div>
                </div>
              )}
              <div className="kiosk-customization__actions">
                <button type="button" className="kiosk-customization__add-more" onClick={addCustomization}>Add to Order</button>
              </div>
            </section>
          </div>
        </div>
      </main>
    );
  }

  if (isCheckingOut) {
    const subtotal = quote?.subtotalMinor ?? 0;
    const tax = quote?.taxAmountMinor ?? 0;
    const total = quote?.totalAmountMinor ?? 0;

    return (
      <main className="kiosk-checkout-screen">
        <section className="kiosk-checkout-summary" aria-labelledby="order-summary-title">
          <header>
            <h1 id="order-summary-title">Order Summary</h1>
            <p>Ticket #10005</p>
          </header>
          <div className="kiosk-checkout-summary__items">
            {cartEntries.map(({ line, index, itemName, modifiers, total: lineTotal }) => (
              <article className="kiosk-checkout-summary__item" key={`${line.menuItemId}-${index}`}>
                <img src={checkoutItemImage} alt="" />
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
              <div><dt>Tax</dt><dd>{formatMoney(tax, menu?.currencyCode)}</dd></div>
              <div className="kiosk-checkout-summary__grand-total"><dt>Total</dt><dd>{formatMoney(total, menu?.currencyCode)}</dd></div>
            </dl>
          </footer>
        </section>
        <section className="kiosk-checkout-payment" aria-labelledby="payment-title">
          {!order ? (
            <button type="button" className="kiosk-checkout-payment__contactless" onClick={() => void checkout()} aria-label="Start payment">
              <img src={checkoutContactless} alt="" />
            </button>
          ) : (
            <div className="kiosk-checkout-payment__scenarios">
              <span>Choose a simulated payment response</span>
              {(['APPROVED', 'DECLINED', 'CANCELLED', 'TIMED_OUT', 'FAILED'] as PaymentScenario[]).map((scenario) => (
                <button type="button" key={scenario} onClick={() => void pay(scenario)}>{scenario.replace('_', ' ')}</button>
              ))}
            </div>
          )}
          <h1 id="payment-title">Payment</h1>
          <p>Tap, insert, or swipe your preferred method below.</p>
          <span className="kiosk-checkout-payment__secure">Secure Encrypted Transaction</span>
          <button type="button" className="kiosk-checkout-payment__cancel" onClick={cancelCheckout}>Cancel Order</button>
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
        <section className="kiosk-upsell" aria-labelledby="upsell-title">
          <h1 id="upsell-title">Complete Your Meal?</h1>
          <p>Perfect pairings to elevate your dining experience.</p>
          <div className="kiosk-upsell__grid">
            {upsellItems.map((item) => (
              <article className="kiosk-upsell__card" key={item.menuItemId}>
                <img className="kiosk-upsell__image" src={item.image} alt={item.alt} />
                <div className="kiosk-upsell__card-content">
                  <h2>{item.name}</h2>
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
      </main>
    );
  }

  return (
    <main className="kiosk-menu-screen">
      <aside className="kiosk-category-nav" aria-label="Menu categories">
        {categoryNavigation.map((category) => {
          const isActive = category.categoryId === activeCategory;
          return (
            <button
              key={category.label}
              type="button"
              className={`kiosk-category-nav__button${isActive ? ' kiosk-category-nav__button--active' : ''}`}
              aria-pressed={isActive}
              onClick={() => setActiveCategory(category.categoryId)}
            >
              <img src={category.icon} alt="" />
              <span>{category.label}</span>
            </button>
          );
        })}
      </aside>
      <section className="kiosk-menu-content" aria-label={`${activeCategoryLabel} menu`}>
        <h1>{activeCategoryLabel}</h1>
        <p className="sr-only" role="status">{message}</p>
        {displayItems.length > 0 ? (
          <div className="kiosk-menu-grid">
            {displayItems.map((item, index) => (
              <article className="kiosk-menu-card" key={`${item.menu_item_id}-${index}`}>
                <img className="kiosk-menu-card__image" src={menuItemImages[item.menu_item_id] ?? classicBurgerImage} alt={item.item_name} />
                <div className="kiosk-menu-card__content">
                  <h2>{item.item_name}</h2>
                  <p>{menuItemDescriptions[item.menu_item_id] ?? 'Made fresh to order.'}</p>
                  <strong>{formatMoney(item.base_price_minor, menu?.currencyCode)}</strong>
                  <button type="button" aria-label={`Customize ${item.item_name}`} onClick={() => openCustomization(item)}>
                    <img src={menuAddIcon} alt="" />
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
        <header className="kiosk-cart__header">
          <h2>Your Order</h2>
          <span>{cart.reduce((sum, line) => sum + line.quantity, 0)} items</span>
        </header>
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
                  <button type="button" onClick={() => changeCartQuantity(index, -1)} aria-label={`Remove one ${itemName}`}>−</button>
                  <span>{line.quantity}</span>
                  <button type="button" onClick={() => changeCartQuantity(index, 1)} aria-label={`Add one ${itemName}`}>+</button>
                </div>
              </li>
            ))}
          </ul>
        ) : (
          <p className="kiosk-cart__empty">Your order is empty. Choose an item to get started.</p>
        )}
        <footer className="kiosk-cart__footer">
          <div className="kiosk-cart__total">
            <span>Total</span>
            <strong>{formatMoney(quote?.totalAmountMinor ?? 0, menu?.currencyCode)}</strong>
          </div>
          {!order ? (
            <button type="button" disabled={!quote} onClick={() => setIsUpselling(true)}>
              Checkout
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
    </main>
  );
}
